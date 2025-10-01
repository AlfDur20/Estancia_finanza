"""
Nearest-Neighbor Contrastive Learning of Representations (NNCLR) implementations.

This module provides implementations of NNCLR for time series embedding,
including CNN and LSTM based architectures.
"""

from .nnclr import NNCLR
from .nnclr_embd import nnclr_cnn_embedding, nnclr_lstm_embedding

__all__ = [
    'NNCLR',
    'nnclr_cnn_embedding',
    'nnclr_lstm_embedding'
]