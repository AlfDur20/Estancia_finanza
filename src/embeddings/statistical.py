"""
Statistical embedding methods for time series data.

This module implements statistical dimensionality reduction techniques including
PCA, ICA, and LDA for time series embedding.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Union, Optional
import warnings


class StatisticalEmbedding:
    """Base class for statistical embedding methods."""
    
    def __init__(self):
        self.is_fitted = False
        self.scaler = None
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be numpy array or pandas DataFrame")
        if X.ndim != 2:
            raise ValueError("Input must be 2-dimensional")
        return X
    
    def _apply_scaling(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
                      X_test: Optional[np.ndarray] = None, 
                      scaling_method: str = 'standard') -> Tuple:
        """Apply scaling to the data."""
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            results.append(X_test_scaled)
        
        return tuple(results) if len(results) > 1 else X_train_scaled


class PCAEmbedding(StatisticalEmbedding):
    """
    Principal Component Analysis embedding for time series data.
    
    Parameters
    ----------
    n_components : int or float, default=None
        Number of components to keep. If float, represents the percentage of variance to retain.
    scaling_method : str, default='standard'
        Scaling method to apply before PCA ('standard' or 'minmax').
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_components: Optional[Union[int, float]] = None, 
                 scaling_method: str = 'standard', random_state: Optional[int] = None):
        super().__init__()
        self.n_components = n_components
        self.scaling_method = scaling_method
        self.random_state = random_state
        self.pca = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'PCAEmbedding':
        """
        Fit the PCA embedding to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : PCAEmbedding
            Fitted estimator.
        """
        X = self._validate_input(X)
        
        # Apply scaling
        X_scaled = self._apply_scaling(X, scaling_method=self.scaling_method)
        
        # Initialize and fit PCA
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
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
            raise ValueError("PCAEmbedding must be fitted before transform")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Fit PCA and transform data."""
        return self.fit(X).transform(X)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if not self.is_fitted:
            raise ValueError("PCAEmbedding must be fitted first")
        return self.pca.explained_variance_ratio_


class LDAEmbedding(StatisticalEmbedding):
    """
    Linear Discriminant Analysis embedding for time series data.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If None, min(n_classes-1, n_features).
    scaling_method : str, default='standard'
        Scaling method to apply before LDA ('standard' or 'minmax').
    solver : str, default='svd'
        Solver to use ('svd', 'lsqr', 'eigen').
    """
    
    def __init__(self, n_components: Optional[int] = None, 
                 scaling_method: str = 'standard', solver: str = 'svd'):
        super().__init__()
        self.n_components = n_components
        self.scaling_method = scaling_method
        self.solver = solver
        self.lda = None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'LDAEmbedding':
        """
        Fit the LDA embedding to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : LDAEmbedding
            Fitted estimator.
        """
        X = self._validate_input(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Apply scaling
        X_scaled = self._apply_scaling(X, scaling_method=self.scaling_method)
        
        # Initialize and fit LDA
        self.lda = LinearDiscriminantAnalysis(
            n_components=self.n_components, 
            solver=self.solver
        )
        self.lda.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform data using fitted LDA.
        
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
            raise ValueError("LDAEmbedding must be fitted before transform")
        
        X = self._validate_input(X)
        X_scaled = self.scaler.transform(X)
        return self.lda.transform(X_scaled)
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Fit LDA and transform data."""
        return self.fit(X, y).transform(X)


def pca_embedding(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                 test_df: pd.DataFrame, n_components: int) -> Tuple[np.ndarray, ...]:
    """
    Legacy function for PCA embedding (maintained for backward compatibility).
    
    Parameters
    ----------
    train_df, valid_df, test_df : DataFrame
        Training, validation, and test datasets.
    n_components : int
        Number of principal components to keep.
        
    Returns
    -------
    tuple of ndarrays
        Embedded training, validation, and test data.
    """
    warnings.warn(
        "pca_embedding function is deprecated. Use PCAEmbedding class instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    pca = PCAEmbedding(n_components=n_components)
    train_pca = pca.fit_transform(train_df)
    valid_pca = pca.transform(valid_df)
    test_pca = pca.transform(test_df)
    
    return train_pca, valid_pca, test_pca


def std_scaling(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
               test_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Legacy function for standard scaling (maintained for backward compatibility).
    """
    warnings.warn(
        "std_scaling function is deprecated. Use StandardScaler directly or embedding classes.",
        DeprecationWarning,
        stacklevel=2
    )
    
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    valid_scaled = pd.DataFrame(scaler.transform(valid_df), columns=valid_df.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    
    return train_scaled, valid_scaled, test_scaled


def minmax_scaling(train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                  test_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """
    Legacy function for minmax scaling (maintained for backward compatibility).
    """
    warnings.warn(
        "minmax_scaling function is deprecated. Use MinMaxScaler directly or embedding classes.",
        DeprecationWarning,
        stacklevel=2
    )
    
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    valid_scaled = pd.DataFrame(scaler.transform(valid_df), columns=valid_df.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    
    return train_scaled, valid_scaled, test_scaled