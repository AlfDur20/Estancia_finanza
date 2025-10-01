"""
CNN Encoder for NNCLR Time Series Embedding
"""

import torch
import torch.nn as nn


class CNN_Encoder(nn.Module):
    """
    CNN-based encoder for time series embedding.
    
    Args:
        input_size (int): The number of features or channels in the input time series.
        embedding_dim (int): The desired dimension of the output embedding.
        sequence_length (int): Length of the input time series.
    """
    
    def __init__(self, input_size: int, embedding_dim: int, sequence_length: int):
        super(CNN_Encoder, self).__init__()
        
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        
        # CNN layers
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=100,
            kernel_size=10,
            padding=4
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=100,
            out_channels=100,
            kernel_size=10,
            padding=4
        )
        
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate the output size after convolutions and pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(100, embedding_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        # x shape: (batch_size, sequence_length, input_size)
        # Transpose for Conv1d: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # First convolutional layer
        x = torch.relu(self.conv1(x))
        
        # Second convolutional layer
        x = torch.relu(self.conv2(x))
        
        # Dropout for regularization
        x = self.dropout(x)
        
        # Max pooling
        x = self.maxpool(x)
        
        # Adaptive pooling to get fixed size output
        x = self.adaptive_pool(x)  # (batch_size, 100, 1)
        
        # Flatten
        x = x.squeeze(-1)  # (batch_size, 100)
        
        # Final projection
        embedding = self.projection(x)  # (batch_size, embedding_dim)
        
        return embedding


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    BATCH_SIZE = 32
    SEQ_LENGTH = 151
    NUM_CHANNELS = 3
    EMBEDDING_DIM = 128
    
    # Create encoder
    encoder = CNN_Encoder(
        input_size=NUM_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=SEQ_LENGTH
    )
    
    print("CNN Encoder Architecture:")
    print(encoder)
    
    # Test with dummy data
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_CHANNELS)
    
    # Get embeddings
    embeddings = encoder(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output embedding shape: {embeddings.shape}")
    
    # Verify output shape
    assert embeddings.shape == (BATCH_SIZE, EMBEDDING_DIM), f"Expected shape {(BATCH_SIZE, EMBEDDING_DIM)}, got {embeddings.shape}"
    print("✅ CNN Encoder test passed!")