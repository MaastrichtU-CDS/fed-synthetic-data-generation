"""
Test configuration and fixtures for fed-synthetic-data testing suite.

This module provides common fixtures and utilities used across all test modules.
It includes synthetic data generation and shared test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List
from unittest.mock import MagicMock
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def sample_tabular_data():
    """
    Generate a sample tabular dataset for testing.

    Returns:
        pd.DataFrame: DataFrame with various numerical and categorical variables.
    """
    np.random.seed(42)  # For reproducible tests

    n_samples = 1000
    data = {
        "age": np.random.normal(45, 15, n_samples).astype(int),
        "height": np.random.normal(170, 10, n_samples),
        "weight": np.random.normal(70, 15, n_samples),
        "income": np.random.lognormal(10, 1, n_samples),
        "gender": np.random.choice(["Male", "Female", "Other"], n_samples, p=[0.48, 0.48, 0.04]),
        "category": np.random.choice(["A", "B", "C"], n_samples),
    }

    # Add some missing values
    data["height"][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data["weight"][np.random.choice(n_samples, 30, replace=False)] = np.nan

    df = pd.DataFrame(data)
    
    # Convert categorical columns
    df["gender"] = df["gender"].astype("category")
    df["category"] = df["category"].astype("category")

    return df


@pytest.fixture
def sample_numerical_data():
    """
    Generate a sample numerical dataset for testing statistical functions.

    Returns:
        pd.DataFrame: DataFrame with various numerical variables.
    """
    np.random.seed(42)

    n_samples = 1000
    data = {
        "feature_1": np.random.normal(50, 10, n_samples),
        "feature_2": np.random.exponential(2, n_samples),
        "feature_3": np.random.uniform(0, 100, n_samples),
        "feature_4": np.random.gamma(2, 2, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_categorical_data():
    """
    Generate a sample categorical dataset for testing.

    Returns:
        pd.DataFrame: DataFrame with categorical variables.
    """
    np.random.seed(42)

    n_samples = 1000
    data = {
        "color": np.random.choice(["Red", "Blue", "Green"], n_samples),
        "size": np.random.choice(["Small", "Medium", "Large"], n_samples),
        "quality": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.2, 0.5, 0.3]),
    }

    df = pd.DataFrame(data)
    for col in df.columns:
        df[col] = df[col].astype("category")

    return df


@pytest.fixture
def federated_datasets():
    """
    Generate multiple datasets simulating federated nodes.

    Returns:
        List[pd.DataFrame]: List of DataFrames representing different nodes.
    """
    np.random.seed(42)
    
    datasets = []
    for i in range(3):
        n_samples = np.random.randint(500, 1500)
        data = {
            "feature_1": np.random.normal(50 + i * 5, 10, n_samples),
            "feature_2": np.random.exponential(2, n_samples),
            "label": np.random.choice([0, 1], n_samples),
            "node_id": [i] * n_samples,
        }
        datasets.append(pd.DataFrame(data))
    
    return datasets


@pytest.fixture
def edge_case_data():
    """
    Generate edge case datasets for robust testing.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing various edge case datasets.
    """
    datasets = {}

    # Empty dataset
    datasets["empty"] = pd.DataFrame()

    # Single row dataset
    datasets["single_row"] = pd.DataFrame({"value": [42], "category": ["A"]})

    # Single column dataset
    datasets["single_column"] = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

    # All NaN dataset
    datasets["all_nan"] = pd.DataFrame({
        "col1": [np.nan] * 10,
        "col2": [np.nan] * 10,
    })

    # Dataset with extreme values
    datasets["extreme_values"] = pd.DataFrame({
        "tiny": [1e-10, 1e-9, 1e-8],
        "huge": [1e10, 1e11, 1e12],
        "zero": [0, 0, 0],
    })

    return datasets


@pytest.fixture
def mock_generator():
    """
    Create a mock synthetic data generator for testing.

    Returns:
        MagicMock: Mock generator with necessary methods.
    """
    mock_gen = MagicMock()
    mock_gen.is_fitted = True
    mock_gen.sample.return_value = pd.DataFrame({
        "feature_1": np.random.normal(50, 10, 100),
        "feature_2": np.random.exponential(2, 100),
    })
    return mock_gen


@pytest.fixture
def privacy_config():
    """
    Provide sample privacy configuration for testing.

    Returns:
        Dict: Privacy configuration parameters.
    """
    return {
        "epsilon": 1.0,
        "delta": 1e-5,
        "mechanism": "laplace",
        "clip_norm": 1.0,
    }


@pytest.fixture
def federated_config():
    """
    Provide sample federated learning configuration.

    Returns:
        Dict: Federated learning configuration parameters.
    """
    return {
        "num_rounds": 10,
        "aggregation_method": "fedavg",
        "num_nodes": 3,
        "local_epochs": 5,
    }


def assert_dataframes_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    check_dtype: bool = True,
    rtol: float = 1e-5
):
    """
    Custom assertion for comparing DataFrames with numerical tolerance.

    Args:
        df1, df2: DataFrames to compare
        check_dtype: Whether to check data types
        rtol: Relative tolerance for numerical comparisons
    """
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    assert list(df1.columns) == list(df2.columns), \
        f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}"

    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_allclose(
                df1[col].dropna(),
                df2[col].dropna(),
                rtol=rtol
            )
        else:
            pd.testing.assert_series_equal(
                df1[col],
                df2[col],
                check_dtype=check_dtype
            )


def assert_privacy_preserved(
    original_data: pd.DataFrame,
    privatized_data: pd.DataFrame,
    epsilon: float
):
    """
    Assert that privacy has been adequately preserved.

    Args:
        original_data: Original data before privacy measures
        privatized_data: Data after privacy measures
        epsilon: Privacy budget
    """
    # Basic checks - more sophisticated checks would be implemented
    assert original_data.shape == privatized_data.shape
    assert list(original_data.columns) == list(privatized_data.columns)
    
    # Check that data has been modified (noise added)
    for col in original_data.select_dtypes(include=[np.number]).columns:
        original_values = original_data[col].dropna().values
        privatized_values = privatized_data[col].dropna().values
        
        if len(original_values) > 0 and len(privatized_values) > 0:
            # Should not be identical if privacy was applied
            if epsilon < 10:  # Only check if reasonable privacy budget
                assert not np.allclose(original_values, privatized_values), \
                    f"Column {col} appears unchanged after privacy application"
