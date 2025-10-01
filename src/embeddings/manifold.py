"""
Manifold learning embedding methods for time series data.

This module implements non-linear dimensionality reduction techniques including
UMAP, LLE, and other manifold learning methods.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding, TSNE, MDS, Isomap
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union, Optional, Dict, Any
import warnings

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


class ManifoldEmbedding:
    """Base class for manifold embedding methods."""
    
    def __init__(self):
        self.is_fitted = False
        self.scaler = None
        self.embedder = None
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be numpy array or pandas DataFrame")
        if X.ndim != 2:
            raise ValueError("Input must be 2-dimensional")
        return X
    
    def _apply_scaling(self, X_train: np.ndarray, 
                      X_val: Optional[np.ndarray] = None, 
                      X_test: Optional[np.ndarray] = None) -> Tuple:
        """Apply standard scaling to the data."""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else X_train_scaled


class UMAPEmbedding(ManifoldEmbedding):
    """
    UMAP (Uniform Manifold Approximation and Projection) embedding.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions for the embedded space.
    n_neighbors : int, default=15
        Number of neighbors to consider for each point.
    min_dist : float, default=0.1
        Minimum distance between embedded points.
    metric : str, default='euclidean'
        Distance metric to use.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, metric: str = 'euclidean',
                 random_state: Optional[int] = None):
        super().__init__()
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is required but not installed. Install with: pip install umap-learn")
        
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'UMAPEmbedding':
        """
        Fit the UMAP embedding to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : UMAPEmbedding
            Fitted estimator.
        """
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        self.embedder.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted UMAP.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
            
        Returns
        -------
        X_embedded : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        if not self.is_fitted:
            raise ValueError("UMAPEmbedding must be fitted before transform")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        return self.embedder.transform(X_scaled)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit UMAP and transform data."""
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        
        X_embedded = self.embedder.fit_transform(X_scaled)
        self.is_fitted = True
        return X_embedded


class LLEEmbedding(ManifoldEmbedding):
    """
    Locally Linear Embedding for time series data.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    method : str, default='standard'
        LLE method ('standard', 'hessian', 'modified', 'ltsa').
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 5,
                 method: str = 'standard', random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.method = method
        self.random_state = random_state
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'LLEEmbedding':
        """Fit the LLE embedding to training data."""
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = LocallyLinearEmbedding(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            method=self.method,
            random_state=self.random_state
        )
        self.embedder.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted LLE."""
        if not self.is_fitted:
            raise ValueError("LLEEmbedding must be fitted before transform")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        return self.embedder.transform(X_scaled)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit LLE and transform data."""
        return self.fit(X).transform(X)


class TSNEEmbedding(ManifoldEmbedding):
    """
    t-SNE (t-Distributed Stochastic Neighbor Embedding) for time series data.
    
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        Related to the number of nearest neighbors.
    learning_rate : str or float, default='warn'
        Learning rate for optimization.
    n_iter : int, default=1000
        Maximum number of iterations.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: Union[str, float] = 'warn', n_iter: int = 1000,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit t-SNE and transform data.
        
        Note: t-SNE does not support separate fit/transform, only fit_transform.
        """
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
        
        X_embedded = self.embedder.fit_transform(X_scaled)
        self.is_fitted = True
        return X_embedded
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'TSNEEmbedding':
        """Fit method (delegates to fit_transform for t-SNE)."""
        warnings.warn("t-SNE does not support separate fit/transform. Use fit_transform instead.")
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform method (not supported for t-SNE)."""
        raise NotImplementedError("t-SNE does not support transform. Use fit_transform for each dataset.")


class IsomapEmbedding(ManifoldEmbedding):
    """
    Isomap embedding for time series data.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of coordinates for the manifold.
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 5):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'IsomapEmbedding':
        """Fit the Isomap embedding to training data."""
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = Isomap(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors
        )
        self.embedder.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Transform data using fitted Isomap."""
        if not self.is_fitted:
            raise ValueError("IsomapEmbedding must be fitted before transform")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        return self.embedder.transform(X_scaled)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit Isomap and transform data."""
        return self.fit(X).transform(X)


class MDSEmbedding(ManifoldEmbedding):
    """
    Multidimensional Scaling (MDS) embedding for time series data.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in which to immerse the dissimilarities.
    metric : bool, default=True
        If True, perform metric MDS; if False, perform non-metric MDS.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_components: int = 2, metric: bool = True,
                 random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'MDSEmbedding':
        """Fit the MDS embedding to training data."""
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = MDS(
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state
        )
        self.embedder.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit MDS and transform data."""
        X = self._validate_input(X)
        X_scaled = self._apply_scaling(X)
        
        self.embedder = MDS(
            n_components=self.n_components,
            metric=self.metric,
            random_state=self.random_state
        )
        
        X_embedded = self.embedder.fit_transform(X_scaled)
        self.is_fitted = True
        return X_embedded


# Legacy functions for backward compatibility
def umap_embedding(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                  test_df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """
    Legacy function for UMAP embedding (maintained for backward compatibility).
    """
    warnings.warn(
        "umap_embedding function is deprecated. Use UMAPEmbedding class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    embedder = UMAPEmbedding()
    train_umap = embedder.fit_transform(train_df)
    val_umap = embedder.transform(val_df)
    test_umap = embedder.transform(test_df)
    
    return train_umap, val_umap, test_umap


def lle_embedding(train_sc: pd.DataFrame, valid_sc: pd.DataFrame, 
                 test_sc: pd.DataFrame, lle_parameters: Dict[str, Any]) -> Tuple[np.ndarray, ...]:
    """
    Legacy function for LLE embedding (maintained for backward compatibility).
    """
    warnings.warn(
        "lle_embedding function is deprecated. Use LLEEmbedding class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    embedder = LLEEmbedding(
        n_neighbors=lle_parameters.get('n_neighbors', 5),
        n_components=lle_parameters.get('n_components', 2),
        random_state=lle_parameters.get('random_state', None)
    )
    
    train_lle = embedder.fit_transform(train_sc)
    valid_lle = embedder.transform(valid_sc)
    test_lle = embedder.transform(test_sc)
    
    return train_lle, valid_lle, test_lle