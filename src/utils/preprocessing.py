"""
Dataset loader for time series classification.

This module handles downloading, loading, and preprocessing of time series
classification datasets from the UCR/UEA repository.
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.io import arff
import warnings

# Create data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets')
os.makedirs(DATA_DIR, exist_ok=True)

# Dictionary of available datasets with their URLs
AVAILABLE_DATASETS = {
    'ElectricDevices': 'https://timeseriesclassification.com/aeon-toolkit/ElectricDevices.zip',
    'ECG5000': 'https://timeseriesclassification.com/aeon-toolkit/ECG5000.zip',
    'Earthquake': 'https://timeseriesclassification.com/aeon-toolkit/Earthquakes.zip',
    'SelfRegulationSCP1': 'https://timeseriesclassification.com/aeon-toolkit/SelfRegulationSCP1.zip',
    'SharePriceIncrease': 'https://raw.githubusercontent.com/hfawaz/dl-4-tsc/master/archives/SharePriceIncrease.zip',
    'RacketSports': 'https://timeseriesclassification.com/aeon-toolkit/RacketSports.zip',
    'MelbournePedestrian': 'https://timeseriesclassification.com/aeon-toolkit/MelbournePedestrian.zip',
}


def download_dataset(dataset_name, force_download=False):
    """
    Download a dataset from the UCR/UEA repository.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to download
    force_download : bool, default=False
        Whether to force download even if the dataset already exists
        
    Returns:
    --------
    str
        Path to the extracted dataset directory
    """
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset {dataset_name} not found in available datasets")
    
    url = AVAILABLE_DATASETS[dataset_name]
    extract_dir = os.path.join(DATA_DIR, dataset_name)
    
    # Check if dataset already exists
    if os.path.exists(extract_dir) and not force_download:
        print(f"Dataset {dataset_name} already exists at {extract_dir}")
        return extract_dir
    
    # Download and extract the dataset
    print(f"Downloading {dataset_name} from {url}...")
    zip_path = os.path.join(DATA_DIR, f"{dataset_name}.zip")
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        
        # Extract the zip file
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zp:
            zp.extractall(extract_dir)
        
        # Remove the zip file
        os.remove(zip_path)
        print(f"Downloaded and extracted {dataset_name} successfully!")
        
        return extract_dir
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"Error downloading {dataset_name}: {e}")
        raise


def load_arff_data(file_path):
    """
    Load data from an ARFF file.
    
    Parameters:
    -----------
    file_path : str
        Path to the ARFF file
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame containing the data
    """
    try:
        raw, meta = arff.loadarff(file_path)
        df = pd.DataFrame(raw)
        
        # Convert byte strings to regular strings if needed
        for col in df.columns:
            if df[col].dtype == object:  # This typically indicates byte strings
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
        return df
    
    except Exception as e:
        print(f"Error loading ARFF file {file_path}: {e}")
        raise


def load_dataset(dataset_name, normalize=True, force_download=False):
    """
    Load a dataset from the UCR/UEA repository.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    normalize : bool, default=True
        Whether to normalize the data using StandardScaler
    force_download : bool, default=False
        Whether to force download even if the dataset already exists
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test) where X and y are pandas DataFrames/Series
    """
    # Download the dataset if needed
    extract_dir = download_dataset(dataset_name, force_download=force_download)
    
    # Load the train and test data
    train_path = os.path.join(extract_dir, f"{dataset_name}_TRAIN.arff")
    test_path = os.path.join(extract_dir, f"{dataset_name}_TEST.arff")
    
    # Check if the expected files exist
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        # Try to find the files with slightly different names
        files = os.listdir(extract_dir)
        train_files = [f for f in files if 'TRAIN' in f.upper()]
        test_files = [f for f in files if 'TEST' in f.upper()]
        
        if train_files and test_files:
            train_path = os.path.join(extract_dir, train_files[0])
            test_path = os.path.join(extract_dir, test_files[0])
        else:
            raise FileNotFoundError(f"Could not find training and test files for {dataset_name}")
    
    # Load the data
    if train_path.endswith('.arff'):
        train_df = load_arff_data(train_path)
        test_df = load_arff_data(test_path)
    elif train_path.endswith('.txt') or train_path.endswith('.csv'):
        # Try to load as CSV
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)
    else:
        raise ValueError(f"Unsupported file format for {train_path}")
    
    # Determine the target column
    if 'target' in train_df.columns:
        target_column = 'target'
    elif 'class' in train_df.columns:
        target_column = 'class'
    else:
        # Assume the first column is the target for standard UCR/UEA datasets
        target_column = train_df.columns[0]
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    # Convert labels to integers if they are strings
    if y_train.dtype == object:
        label_encoder = LabelEncoder()
        y_train = pd.Series(label_encoder.fit_transform(y_train))
        y_test = pd.Series(label_encoder.transform(y_test))
    
    # Ensure labels are numeric
    y_train = pd.to_numeric(y_train, errors='coerce')
    y_test = pd.to_numeric(y_test, errors='coerce')
    
    # Normalize the data if requested
    if normalize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    # Print dataset information
    print(f"Dataset: {dataset_name}")
    print(f"Number of classes: {len(y_train.unique())}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, y_train, X_test, y_test


def prepare_dataset(dataset_name, valid_size=0.2, normalize=True, random_state=42, force_download=False):
    """
    Load and prepare a dataset for embedding experiments.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset to load
    valid_size : float, default=0.2
        Proportion of the test set to use as validation set
    normalize : bool, default=True
        Whether to normalize the data using StandardScaler
    random_state : int, default=42
        Random seed for reproducibility
    force_download : bool, default=False
        Whether to force download even if the dataset already exists
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_valid, y_valid, X_test, y_test) where X and y are pandas DataFrames/Series
    """
    # Load the dataset
    X_train, y_train, X_test, y_test = load_dataset(
        dataset_name, normalize=normalize, force_download=force_download
    )
    
    # Split the test set into validation and test sets
    if valid_size > 0:
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_test, y_test,
            test_size=0.5,  # Split the test set equally into validation and test
            random_state=random_state,
            stratify=y_test if len(y_test.unique()) > 1 else None
        )
    else:
        X_valid, y_valid = None, None
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def convert_to_channels(X, n_channels=1):
    """
    Convert 2D data to 3D data with channels.
    For multivariate time series, this reshapes the data to (samples, timesteps, channels).
    
    Parameters:
    -----------
    X : DataFrame or array-like
        2D data with shape (samples, features)
    n_channels : int, default=1
        Number of channels in the time series
        
    Returns:
    --------
    array
        3D array with shape (samples, timesteps, channels)
    """
    # Convert DataFrame to numpy array if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    n_samples, n_features = X.shape
    
    # Calculate number of timesteps
    n_timesteps = n_features // n_channels
    
    if n_features % n_channels != 0:
        warnings.warn(f"Number of features ({n_features}) is not divisible by number of channels ({n_channels}). Using {n_timesteps} timesteps.")
    
    # Reshape the data
    if n_channels == 1:
        # For univariate time series, simply add a channel dimension
        return X.reshape(n_samples, n_timesteps, n_channels)
    else:
        # For multivariate time series, need to reorder the data
        # This assumes the data is organized with all channels for timestep 1, then all channels for timestep 2, etc.
        X_3d = np.zeros((n_samples, n_timesteps, n_channels))
        
        for i in range(n_timesteps):
            for j in range(n_channels):
                feature_idx = i * n_channels + j
                if feature_idx < n_features:
                    X_3d[:, i, j] = X[:, feature_idx]
        
        return X_3d


def get_dataset_channels(dataset_name):
    """
    Get the number of channels for a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
        
    Returns:
    --------
    int
        Number of channels in the dataset
    """
    dataset_channels = {
        'ElectricDevices': 1,
        'ECG5000': 1,
        'Earthquake': 1,
        'SelfRegulationSCP1': 6,
        'SharePriceIncrease': 1,
        'RacketSports': 6,
        'MelbournePedestrian': 1,
    }
    
    return dataset_channels.get(dataset_name, 1)  # Default to 1 channel if unknown


if __name__ == "__main__":
    # Example usage
    dataset_name = 'ECG5000'
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_dataset(dataset_name)
    
    # Get number of channels
    n_channels = get_dataset_channels(dataset_name)
    
    # Convert to 3D with channels
    X_train_3d = convert_to_channels(X_train, n_channels)
    X_valid_3d = convert_to_channels(X_valid, n_channels)
    X_test_3d = convert_to_channels(X_test, n_channels)
    
    print(f"\n3D Shapes:")
    print(f"  X_train_3d: {X_train_3d.shape}")
    print(f"  X_valid_3d: {X_valid_3d.shape}")
    print(f"  X_test_3d:  {X_test_3d.shape}")