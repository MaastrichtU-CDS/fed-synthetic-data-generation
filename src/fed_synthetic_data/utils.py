"""
Utility functions for federated synthetic data generation.

This module provides general utility functions used across the library.

TODO: Implement utility functions for data manipulation, logging, and other common tasks.
"""

import base64
import numpy as np
import pandas as pd


def weights_to_json(weights: list[np.ndarray]) -> list[dict]:
    """
    Serialise model weights for JSON transport.

    Args:
        weights (list[np.ndarray]): List of model weights to serialise.

    Returns:
        list[dict]: List of dictionaries representing the weights in JSON-serialisable format.
    """
    return [
        {"shape": w.shape, "dtype": str(w.dtype), "data": base64.b64encode(w.tobytes()).decode()}
        for w in weights
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

    If no column_order is provided, the columns are sorted alphabetically and the
    resulting order can be passed to subsequent nodes to enforce consistency.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_order (list[str] | None): An explicit column order to apply. If None,
            columns are sorted alphabetically.

    Returns:
        pd.DataFrame: A re-indexed DataFrame with columns in the specified order.
    """
    if column_order is None:
        column_order = sorted(df.columns)
    return df.reindex(column_order, axis=1)
