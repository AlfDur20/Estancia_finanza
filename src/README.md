# Source Code

This directory contains the modular implementation of time series embedding methods and classification algorithms from our paper "Time Series Embedding Methods for Classification Tasks: A Review".

## Directory Structure
```bash
src/
├── classifiers/                 # Classification algorithms
├── datasets/                    # Dataset loading and preprocessing
├── embeddings/                  # Embedding method implementations
│   ├── statistical.py           # PCA, ICA, LDA
│   ├── manifold.py              # t-SNE, UMAP, MDS, LLE
│   ├── nnclr/                   # Transformer, CNNs, RNNs with NNCLR
│   └── time_series_embedding.py # Traditional time series embedding methods
└── utils/                       # Utility functions and helpers
```

## Utils Directory
Supporting functions and utilities:

- Augmentation
- Helper functions for data processing
- Visualization tools
- Configuration management


## 📊 Evaluation Pipeline
The classification evaluation follows this pattern:
Time Series Data → Embedding Method → Feature Vector → Classifier → Results

- Load data using datasets/ utilities
- Apply embedding using methods from embeddings/
- Evaluate using classifiers from classifiers/
- Analyze results with tools from utils/


Note: This implementation accompanies our research paper. For detailed methodology and experimental results, please refer to the main paper: arXiv:2501.13392