"""Unit tests for data loader module."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import sqlite3
from src.data_loader import remove_unit_suffix, split_sequence_multi_output


def test_remove_unit_suffix():
    """Test unit suffix removal function."""
    assert remove_unit_suffix("128Mi") == 128.0
    assert remove_unit_suffix("700m") == 700.0
    assert remove_unit_suffix("2Gi") == 2.0
    assert remove_unit_suffix(100) == 100
    assert remove_unit_suffix("test") == "test"


def test_split_sequence_multi_output():
    """Test sequence splitting for multi-step forecasting."""
    sequence = np.arange(100)
    n_steps = 10
    n_future = 5
    X, y = split_sequence_multi_output(sequence, n_steps, n_future)

    # Check shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == n_steps
    assert y.shape[1] == n_future

    # Check first sample
    assert np.array_equal(X[0], np.arange(10))
    assert np.array_equal(y[0], np.arange(10, 15))


def test_load_and_preprocess_data():
    """Test data loading from SQLite with dummy data."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create test database
        conn = sqlite3.connect(db_path)
        n = 500
        data = {
            "Timestamp": pd.date_range("2026-05-09", periods=n, freq="1s").astype(str),
            "cpu_usage_firewall": np.random.rand(n) * 100,
            "traffic_rate_firewall": np.random.rand(n) * 1000,
            "processing_delay_firewall": np.random.rand(n) * 10,
            "cpu_usage_dpi": np.random.rand(n) * 100,
            "traffic_rate_dpi": np.random.rand(n) * 1000,
            "processing_delay_dpi": np.random.rand(n) * 10,
            "cpu_usage_enc": np.random.rand(n) * 100,
            "traffic_rate_enc": np.random.rand(n) * 1000,
            "processing_delay_enc": np.random.rand(n) * 10,
            "cpu_usage_comp": np.random.rand(n) * 100,
            "traffic_rate_comp": np.random.rand(n) * 1000,
            "processing_delay_comp": np.random.rand(n) * 10,
            "memory_usage_dpi": np.random.rand(n) * 1000,
        }
        pd.DataFrame(data).to_sql("VNF_KPI_database", conn, if_exists="replace", index=False)
        conn.close()

        # Load config
        config = {
            "data": {
                "db_path": db_path,
                "table_name": "VNF_KPI_database",
                "train_ratio": 0.8,
                "n_steps": 10,
                "n_future_steps": 5,
                "features": [
                    "cpu_usage_firewall",
                    "traffic_rate_firewall",
                    "processing_delay_firewall",
                    "cpu_usage_dpi",
                    "traffic_rate_dpi",
                    "processing_delay_dpi",
                    "cpu_usage_enc",
                    "traffic_rate_enc",
                    "processing_delay_enc",
                    "cpu_usage_comp",
                    "traffic_rate_comp",
                    "processing_delay_comp",
                    "memory_usage_dpi",
                ],
            }
        }

        from src.data_loader import load_and_preprocess_data

        data = load_and_preprocess_data(config)

        # Verify outputs
        assert "X_train" in data
        assert "y_train" in data
        assert "X_test" in data
        assert "y_test" in data
        assert data["n_features"] == len(config["data"]["features"])
        assert data["n_steps"] == 10
        assert data["n_future_steps"] == 5

    finally:
        os.unlink(db_path)