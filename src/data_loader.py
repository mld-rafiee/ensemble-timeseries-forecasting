"""Data loading and preprocessing module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import get_project_root
import os

def remove_unit_suffix(value):
    """Remove unit suffixes from values (e.g., '128Mi' -> 128)."""
    if isinstance(value, str):
        for suffix in ["m", "Mi", "Gi"]:
            if value.endswith(suffix):
                return float(value[:-len(suffix)])
    return value

def split_sequence_multi_output(sequence, n_steps, n_future_steps):
    """Split sequence into input and multi-step output."""
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        future_end_ix = end_ix + n_future_steps
        if future_end_ix > len(sequence):
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix:future_end_ix])
    return np.array(X), np.array(y)

def load_and_preprocess_data(config):
    """Load and preprocess the VNF KPI dataset."""
    project_root = get_project_root()
    file_path = os.path.join(project_root, config['data']['raw_path'])
    
    # Load data
    data = pd.read_csv(file_path)
    
    # Remove unit suffixes
    data = data.applymap(remove_unit_suffix)
    
    # Select features
    features = config['data']['features']
    data = data[features]
    
    # Remove rows with NaN
    data = data.dropna()
    
    # Split into train/test
    n_samples = data.shape[0]
    train_size = int(n_samples * config['data']['train_ratio'])
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Normalize
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    # Create sequences
    n_steps = config['data']['n_steps']
    n_future = config['data']['n_future_steps']
    n_features = len(features)
    
    X_train, y_train = split_sequence_multi_output(train_scaled, n_steps, n_future)
    X_test, y_test = split_sequence_multi_output(test_scaled, n_steps, n_future)
    
    # Reshape to (samples, n_steps, n_features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': features,
        'n_features': n_features,
        'n_steps': n_steps,
        'n_future_steps': n_future
    }