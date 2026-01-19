"""
Utility functions for federated synthetic data generation.

This module provides general utility functions used across the library.
"""

from typing import Any, Optional, Union, List
import logging
import pandas as pd
import numpy as np


# Configure logging
logger = logging.getLogger(__name__)


def safe_log(level: str, message: str, **kwargs) -> None:
    """
    Safely log a message with the specified level.
    
    This function provides a consistent logging interface across the library
    and handles potential logging errors gracefully.
    
    Args:
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        message: Message to log
        **kwargs: Additional logging parameters
    """
    level = level.lower()
    
    log_functions = {
        'debug': logger.debug,
        'info': logger.info,
        'warning': logger.warning,
        'error': logger.error,
        'critical': logger.critical,
    }
    
    log_func = log_functions.get(level, logger.info)
    
    try:
        log_func(message, **kwargs)
    except Exception as e:
        # Fallback to print if logging fails
        print(f"[{level.upper()}] {message}")


def validate_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> bool:
    """
    Validate that data meets basic requirements.
    
    Args:
        data: Data to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if data is valid
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    
    if len(data) < min_rows:
        raise ValueError(f"Data must have at least {min_rows} rows, got {len(data)}")
    
    if required_columns:
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def check_column_types(
    data: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None
) -> dict:
    """
    Check and categorize column types in a DataFrame.
    
    Args:
        data: DataFrame to analyze
        categorical_columns: Optional list of categorical column names
        numerical_columns: Optional list of numerical column names
        
    Returns:
        Dictionary with categorized column information
    """
    if categorical_columns is None:
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numerical_columns is None:
        numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    return {
        'categorical': categorical_columns,
        'numerical': numerical_columns,
        'all_columns': data.columns.tolist(),
        'n_categorical': len(categorical_columns),
        'n_numerical': len(numerical_columns),
    }


def compute_data_statistics(data: pd.DataFrame) -> dict:
    """
    Compute basic statistics about a dataset.
    
    Args:
        data: DataFrame to analyze
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'n_rows': len(data),
        'n_columns': len(data.columns),
        'n_missing': data.isna().sum().sum(),
        'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Add column type information
    column_types = check_column_types(data)
    stats.update(column_types)
    
    return stats


def split_data_by_nodes(
    data: pd.DataFrame,
    n_nodes: int,
    method: str = "iid",
    random_state: Optional[int] = None
) -> List[pd.DataFrame]:
    """
    Split data across multiple nodes for federated learning.
    
    Args:
        data: Data to split
        n_nodes: Number of nodes to split data across
        method: Splitting method ('iid' for random, 'non-iid' for label-based)
        random_state: Random seed for reproducibility
        
    Returns:
        List of DataFrames, one per node
    """
    if n_nodes <= 0:
        raise ValueError("Number of nodes must be positive")
    
    if n_nodes > len(data):
        raise ValueError("Number of nodes cannot exceed number of samples")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    if method == "iid":
        # Random IID split
        indices = np.random.permutation(len(data))
        splits = np.array_split(indices, n_nodes)
        return [data.iloc[split].reset_index(drop=True) for split in splits]
    
    elif method == "non-iid":
        # Non-IID split (placeholder - would need label information)
        # For now, just do sequential split as placeholder
        splits = np.array_split(range(len(data)), n_nodes)
        return [data.iloc[split].reset_index(drop=True) for split in splits]
    
    else:
        raise ValueError(f"Unknown splitting method: {method}")


def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple datasets into a single DataFrame.
    
    Args:
        datasets: List of DataFrames to merge
        
    Returns:
        Merged DataFrame
    """
    if not datasets:
        raise ValueError("No datasets provided")
    
    return pd.concat(datasets, ignore_index=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
        return default
    
    result = numerator / denominator
    
    if np.isnan(result) or np.isinf(result):
        return default
    
    return result


def normalize_weights(weights: List[float]) -> List[float]:
    """
    Normalize a list of weights to sum to 1.
    
    Args:
        weights: List of weight values
        
    Returns:
        Normalized weights
    """
    if not weights:
        raise ValueError("Weights list cannot be empty")
    
    total = sum(weights)
    
    if total <= 0:
        # Equal weights if sum is invalid
        return [1.0 / len(weights)] * len(weights)
    
    return [w / total for w in weights]
