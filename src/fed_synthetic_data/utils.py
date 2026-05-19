"""
Utility functions for federated synthetic data generation.

This module provides general utility functions used across the library.

TODO: Implement utility functions for data manipulation, logging, and other common tasks.
"""

import base64
import numpy as np
import pandas as pd
from typing import Any


def _to_numpy(w: Any) -> np.ndarray:
    """
    Convert a weight array (numpy or torch.Tensor) to numpy array.

    Args:
        w: Weight array - can be numpy.ndarray or torch.Tensor.

    Returns:
        numpy.ndarray: The weight as a numpy array.
    """
    # Check if it's a torch Tensor
    if hasattr(w, "detach"):
        w = w.detach().cpu().numpy()
    return np.asarray(w)


def weights_to_json(weights: list[np.ndarray]) -> list[dict]:
    """
    Serialise model weights for JSON transport.

    Args:
        weights (list[np.ndarray]): List of model weights to serialise.
            Can be numpy arrays or PyTorch tensors (as returned by fed-mostlyai-engine).

    Returns:
        list[dict]: List of dictionaries representing the weights in JSON-serialisable format.
    """
    return [
        {"shape": w.shape, "dtype": str(w.dtype), "data": base64.b64encode(w.tobytes()).decode()}
        for w in (_to_numpy(wt) for wt in weights)
    ]


def weights_from_json(entries: list[dict]) -> list[np.ndarray]:
    """
    Deserialise model weights from JSON transport format.

    Each dictionary must contain:
        - "data": Base64-encoded weight data as a string.
        - "dtype": A string representing the data type of the weight.
        - "shape": A tuple (or list) representing the shape of the weight array.

    Args:
        entries (list[dict]): List of serialised weight dictionaries.

    Returns:
        list[np.ndarray]: Deserialised model weights as numpy arrays.
    """
    return [
        np.frombuffer(base64.b64decode(e["data"]), dtype=e["dtype"]).reshape(e["shape"])
        for e in entries
    ]


def sort_columns(df: pd.DataFrame, column_order: list[str] | None = None) -> pd.DataFrame:
    """
    Sort the columns of a DataFrame alphabetically, or reorder them according to a
    provided column order.

    Ensures a consistent column order across federated nodes. This is required
    when flexible_generation is disabled in TabularARGN, which enforces a strict
    column order at generation time.

    If no column_order is provided, the columns are sorted with numeric types first
    (sorted numerically), then string types (sorted alphabetically). This handles
    mixed-type column names safely. For duplicate column names, uses iloc-based
    reordering to avoid pandas errors.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_order (list[str] | None): An explicit column order to apply. If None,
            columns are sorted with numbers first, then strings.

    Returns:
        pd.DataFrame: A re-indexed DataFrame with columns in the specified order.

    Raises:
        ValueError: If column_order is provided but contains columns not in df.
    """
    if column_order is None:
        # Handle mixed types by sorting: numbers first (by value), then strings (alphabetically)
        columns = list(df.columns)

        def sort_key(col):
            # Numbers come first (group 0), strings come second (group 1)
            if isinstance(col, (int, float, np.integer, np.floating)):
                return (0, float(col))
            else:
                return (1, str(col))

        column_order = sorted(columns, key=sort_key)

    # Check for duplicate columns - if present, we need to use iloc-based reordering
    if len(set(column_order)) < len(column_order):
        # Has duplicates - find the permutation indices
        col_list = list(df.columns)
        # Build a mapping from column name to positions in column_order
        # For duplicates, we maintain their relative order
        order_indices = []
        for col in column_order:
            # Find the first occurrence of this column in the original df that hasn't been used yet
            for i, orig_col in enumerate(col_list):
                if orig_col == col and i not in order_indices:
                    order_indices.append(i)
                    break
        return df.iloc[:, order_indices]

    return df.reindex(column_order, axis=1)
