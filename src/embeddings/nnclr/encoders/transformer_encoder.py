"""
Transformer Encoder for NNCLR Time Series Embedding
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def pos_encoding(self, q_len, d_model, normalize=True):
        pe = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if normalize:
            pe = pe - pe.mean()
            pe = pe / (pe.std() * 10)
        return pe

    def forward(self, x):
        x = x + self.pos_encoding(q_len=x.size(1), d_model=x.size(2))
        return self.dropout(x)


class TimeSeriesPatchEmbeddingLayer(nn.Module):
    """Patch embedding layer for transformer encoder."""
    
    def __init__(self, in_channels, patch_size, embedding_dim, input_timesteps):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        # Calculate the number of patches, adjusting for padding if necessary
        # Ceiling division to account for padding
        self.num_patches = -(-input_timesteps // patch_size)
        self.padding = (
            self.num_patches * patch_size
        ) - input_timesteps  # Calculate padding length

        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token_embeddings = nn.Parameter(
            torch.randn((1, 1, embedding_dim), requires_grad=True)
        )
        self.position_embeddings = PositionalEncoding(embedding_dim, dropout=0.1, max_len=input_timesteps)

    def forward(self, x):
        # Pad the input sequence if necessary
        if self.padding > 0:
            x = nn.functional.pad(x, (0, 0, 0, self.padding))  # Pad the second to last dimension, which is input_timesteps

        # We use a Conv1d layer to generate the patch embeddings
        x = x.permute(0, 2, 1)  # (batch, features, timesteps)
        conv_output = self.conv_layer(x)
        conv_output = conv_output.permute(0, 2, 1)  # (batch, timesteps, features)

        batch_size = x.shape[0]
        class_tokens = self.class_token_embeddings.expand(batch_size, -1, -1)
        output = torch.cat((class_tokens, conv_output), dim=1)

        output = self.position_embeddings(output)

        return output


class Transformer_Encoder(nn.Module):
    """
    Transformer encoder for time series data.
    
    Args:
        input_size (int): Number of features in the input time series.
        embedding_dim (int): Dimension of the embedding space.
        sequence_length (int): Length of the input time series.
        patch_size (int): Size of patches for patch embedding.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dim_feedforward (int): Dimension of feedforward network.
        dropout (float): Dropout rate.
    """
    
    def __init__(self, 
                 input_size: int, 
                 embedding_dim: int, 
                 sequence_length: int,
                 patch_size: int = 8, 
                 num_heads: int = 8, 
                 num_layers: int = 6, 
                 dim_feedforward: int = 256, 
                 dropout: float = 0.1):
        super(Transformer_Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embedding = TimeSeriesPatchEmbeddingLayer(
            in_channels=input_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            input_timesteps=sequence_length
        )
        
        # Calculate the number of patches
        self.num_patches = -(-sequence_length // patch_size)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the transformer encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        # x shape: (batch_size, sequence_length, input_size)
        
        # Get patch embeddings with class token
        x = self.patch_embedding(x)  # (batch_size, num_patches + 1, embedding_dim)
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, num_patches + 1, embedding_dim)
        
        # Use class token for final embedding
        class_token_output = x[:, 0, :]  # (batch_size, embedding_dim)
        
        # Final projection
        embedding = self.output_projection(class_token_output)
        
        return embedding


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    BATCH_SIZE = 32
    SEQ_LENGTH = 151
    NUM_CHANNELS = 3
    EMBEDDING_DIM = 128
    PATCH_SIZE = 8

    # Create transformer encoder
    encoder = Transformer_Encoder(
        input_size=NUM_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=SEQ_LENGTH,
        patch_size=PATCH_SIZE,
        num_heads=4,
        num_layers=4,
        dim_feedforward=128,
        dropout=0.2
    )

    print("Transformer Encoder Architecture:")
    print(encoder)

    # Test with dummy data
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_CHANNELS)

    # Get embeddings
    embeddings = encoder(dummy_input)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output embedding shape: {embeddings.shape}")
    
    # Verify output shape
    assert embeddings.shape == (BATCH_SIZE, EMBEDDING_DIM), f"Expected shape {(BATCH_SIZE, EMBEDDING_DIM)}, got {embeddings.shape}"
    print("✅ Transformer Encoder test passed!")
    
    # Test patch embedding layer separately
    patch_embedding_layer = TimeSeriesPatchEmbeddingLayer(
        in_channels=NUM_CHANNELS,
        patch_size=PATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        input_timesteps=SEQ_LENGTH,
    )

    patch_embeddings = patch_embedding_layer(dummy_input)
    expected_patches = -(-SEQ_LENGTH // PATCH_SIZE) + 1  # +1 for class token
    
    print(f"\nPatch embeddings shape: {patch_embeddings.shape}")
    print(f"Expected patches (including class token): {expected_patches}")
    
    assert patch_embeddings.shape == (BATCH_SIZE, expected_patches, EMBEDDING_DIM)
    print("✅ Patch Embedding Layer test passed!")