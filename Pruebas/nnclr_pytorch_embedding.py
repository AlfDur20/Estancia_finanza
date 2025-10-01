"""
PyTorch NNCLR Embedding Module

A simplified PyTorch implementation of Nearest Neighbor Contrastive Learning of Representations (NNCLR)
for generating high-quality embeddings from time series data.

Supports three encoder types: CNN, LSTM, and Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

# Import the modular encoders
from cnn_encoder import CNN_Encoder
from lstm_encoder import LSTM_Encoder
from transformer_encoder import Transformer_Encoder


class Augmenter(nn.Module):
    """Simple augmentation module for time series data."""
    
    def __init__(self, noise_factor=0.1, dropout_rate=0.1):
        super(Augmenter, self).__init__()
        self.noise_factor = noise_factor
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, training=True):
        if training:
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_factor
            x_aug = x + noise
            
            # Apply dropout
            x_aug = self.dropout(x_aug)
            
            return x_aug
        return x


class NNCLR_Embedding(nn.Module):
    """
    NNCLR model for generating embeddings from time series data.
    """
    
    def __init__(
        self,
        input_shape,
        embedding_dim=128,
        encoder_type='cnn',
        temperature=0.1,
        queue_size=5000,
        n_classes=None
    ):
        super(NNCLR_Embedding, self).__init__()
        
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.n_classes = n_classes
        
        sequence_length, n_features = input_shape
        
        # Create encoder based on type
        if encoder_type == 'cnn':
            self.encoder = CNN_Encoder(
                input_size=n_features,
                embedding_dim=embedding_dim,
                sequence_length=sequence_length
            )
        elif encoder_type == 'lstm':
            self.encoder = LSTM_Encoder(
                input_size=n_features,
                embedding_dim=embedding_dim,
                hidden_size=128
            )
        elif encoder_type == 'transformer':
            self.encoder = Transformer_Encoder(
                input_size=n_features,
                embedding_dim=embedding_dim,
                sequence_length=sequence_length,
                patch_size=8,
                num_heads=8,
                num_layers=6,
                dim_feedforward=256,
                dropout=0.1
            )
        else:
            raise ValueError("encoder_type must be 'cnn', 'lstm', or 'transformer'")
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Linear probe for classification (if needed)
        if n_classes is not None:
            self.linear_probe = nn.Linear(embedding_dim, n_classes)
        
        # Augmenter
        self.augmenter = Augmenter()
        
        # Feature queue for nearest neighbor search
        self.register_buffer(
            'feature_queue',
            torch.randn(queue_size, embedding_dim).float()
        )
        self.feature_queue = nn.functional.normalize(self.feature_queue, dim=1)
        
    def nearest_neighbor(self, projections):
        """Find nearest neighbors in the feature queue."""
        # Normalize projections
        projections = nn.functional.normalize(projections, dim=1)
        
        # Compute similarities with queue
        similarities = torch.matmul(projections, self.feature_queue.t())
        
        # Get nearest neighbors
        _, nn_indices = similarities.max(dim=1)
        nn_projections = self.feature_queue[nn_indices]
        
        # Return projections + stop_gradient(nn_projections - projections)
        return projections + (nn_projections - projections).detach()
    
    def contrastive_loss(self, z1, z2):
        """Compute contrastive loss with nearest neighbors."""
        batch_size = z1.size(0)
        
        # Normalize projections
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # Get nearest neighbors
        nn_z1 = self.nearest_neighbor(z1)
        nn_z2 = self.nearest_neighbor(z2)
        
        # Compute similarities
        sim_1_2_1 = torch.matmul(nn_z1, z2.t()) / self.temperature
        sim_1_2_2 = torch.matmul(z2, nn_z1.t()) / self.temperature
        sim_2_1_1 = torch.matmul(nn_z2, z1.t()) / self.temperature
        sim_2_1_2 = torch.matmul(z1, nn_z2.t()) / self.temperature
        
        # Create labels
        labels = torch.arange(batch_size).to(z1.device)
        
        # Compute cross-entropy loss
        loss = (
            nn.functional.cross_entropy(sim_1_2_1, labels) +
            nn.functional.cross_entropy(sim_1_2_2, labels) +
            nn.functional.cross_entropy(sim_2_1_1, labels) +
            nn.functional.cross_entropy(sim_2_1_2, labels)
        ) / 4
        
        # Update feature queue
        self._update_queue(z1)
        
        return loss
    
    def _update_queue(self, features):
        """Update the feature queue with new features."""
        batch_size = features.size(0)
        
        # Normalize features
        features = nn.functional.normalize(features, dim=1)
        
        # Update queue
        self.feature_queue = torch.cat([
            features.detach(),
            self.feature_queue[:-batch_size]
        ], dim=0)
    
    def get_embeddings(self, x):
        """Get embeddings from the encoder."""
        self.eval()
        with torch.no_grad():
            if len(x.shape) == 2:
                x = x.unsqueeze(-1)
            
            # All encoders expect (batch, seq_len, features)
            embeddings = self.encoder(x)
            return embeddings
    
    def forward(self, x1, x2=None):
        """Forward pass for training or inference."""
        if len(x1.shape) == 2:
            x1 = x1.unsqueeze(-1)
        
        if self.training and x2 is not None:
            # Training mode - contrastive learning
            if len(x2.shape) == 2:
                x2 = x2.unsqueeze(-1)
            
            # Get embeddings
            h1 = self.encoder(x1)
            h2 = self.encoder(x2)
            
            # Get projections
            z1 = self.projection_head(h1)
            z2 = self.projection_head(h2)
            
            # Compute contrastive loss
            loss = self.contrastive_loss(z1, z2)
            
            return loss, h1, h2
        else:
            # Inference mode - just return embeddings
            return self.get_embeddings(x1)


def nnclr_embedding_base(
    x_train, x_val, x_test, y_train, y_val, y_test,
    embedding_dim=128, encoder_type='cnn', n_classes=None,
    batch_size=32, pretrain_epochs=20, learning_rate=1e-3,
    device=None, return_timing=False
):
    """
    Base function to generate embeddings using NNCLR.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to numpy if pandas DataFrame
    if hasattr(x_train, 'values'):
        x_train = x_train.values
    if hasattr(x_val, 'values'):
        x_val = x_val.values
    if hasattr(x_test, 'values'):
        x_test = x_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_val, 'values'):
        y_val = y_val.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Reshape if needed
    y_train = y_train.reshape(-1) if len(y_train.shape) > 1 else y_train
    y_val = y_val.reshape(-1) if len(y_val.shape) > 1 else y_val
    y_test = y_test.reshape(-1) if len(y_test.shape) > 1 else y_test
    
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Determine input shape
    input_shape = x_train.shape[1:]
    if len(input_shape) == 1:
        input_shape = input_shape + (1,)
    
    # Start timing
    if return_timing:
        train_time = time.time() - start_train
        start_inference = time.time()
    
    # Generate embeddings
    model.eval()
    print("Generating embeddings...")
    
    with torch.no_grad():
        # Convert to tensors and move to device
        x_train_tensor = torch.FloatTensor(x_train).to(device)
        x_val_tensor = torch.FloatTensor(x_val).to(device)
        x_test_tensor = torch.FloatTensor(x_test).to(device)
        
        # Generate embeddings
        train_embeddings = []
        for i in range(0, len(x_train_tensor), batch_size):
            batch = x_train_tensor[i:i+batch_size]
            emb = model(batch)
            train_embeddings.append(emb.cpu())
        train_embeddings = torch.cat(train_embeddings, dim=0)
        
        val_embeddings = []
        for i in range(0, len(x_val_tensor), batch_size):
            batch = x_val_tensor[i:i+batch_size]
            emb = model(batch)
            val_embeddings.append(emb.cpu())
        val_embeddings = torch.cat(val_embeddings, dim=0)
        
        test_embeddings = []
        for i in range(0, len(x_test_tensor), batch_size):
            batch = x_test_tensor[i:i+batch_size]
            emb = model(batch)
            test_embeddings.append(emb.cpu())
        test_embeddings = torch.cat(test_embeddings, dim=0)
    
    # Convert to pandas DataFrames
    train_emb_df = pd.DataFrame(train_embeddings.numpy())
    val_emb_df = pd.DataFrame(val_embeddings.numpy())
    test_emb_df = pd.DataFrame(test_embeddings.numpy())
    
    if return_timing:
        inference_time = time.time() - start_inference
        
        print(f"NNCLR {encoder_type.upper()} Training time: {train_time:.4f} seconds")
        print(f"NNCLR {encoder_type.upper()} Inference time: {inference_time:.4f} seconds")
        
        return train_emb_df, val_emb_df, test_emb_df, train_time, inference_time
    
    return train_emb_df, val_emb_df, test_emb_df


# The three specific functions you requested
def nnclr_cnn_embedding_with_timing(x_train_df, x_val_df, x_test_df, 
                                   y_train_df, y_val_df, y_test_df, 
                                   width, n_classes):
    """CNN-based NNCLR embedding with timing - matches TensorFlow API."""
    return nnclr_embedding_base(
        x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df,
        embedding_dim=width, encoder_type='cnn', n_classes=n_classes,
        return_timing=True
    )


def nnclr_lstm_embedding_with_timing(x_train_df, x_val_df, x_test_df,
                                    y_train_df, y_val_df, y_test_df,
                                    width, n_classes):
    """LSTM-based NNCLR embedding with timing - matches TensorFlow API."""
    return nnclr_embedding_base(
        x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df,
        embedding_dim=width, encoder_type='lstm', n_classes=n_classes,
        return_timing=True
    )


def nnclr_transformer_embedding_with_timing(x_train_df, x_val_df, x_test_df,
                                           y_train_df, y_val_df, y_test_df,
                                           width, n_classes):
    """Transformer-based NNCLR embedding with timing."""
    return nnclr_embedding_base(
        x_train_df, x_val_df, x_test_df, y_train_df, y_val_df, y_test_df,
        embedding_dim=width, encoder_type='transformer', n_classes=n_classes,
        return_timing=True
    )


