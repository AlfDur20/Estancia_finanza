"""
Classification algorithms for time series embeddings.

This package provides various classification algorithms optimized
for time series embedding vectors, including traditional machine
learning models and neural network approaches.
"""

from .traditional_models import (
    optimize_LOGRG,
    optimize_DT,
    optimize_RF,
    optimize_KNN,
    optimize_XGBOOST,
    optimize_SVM
)

from .neural_models import (
    optimize_MLP,
    optimize_NB
)

__all__ = [
    'optimize_LOGRG',
    'optimize_DT',
    'optimize_RF',
    'optimize_KNN',
    'optimize_XGBOOST',
    'optimize_SVM',
    'optimize_MLP',
    'optimize_NB'
]