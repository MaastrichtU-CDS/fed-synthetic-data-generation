"""
Utility functions for federated synthetic data generation.

This module provides general utility functions used across the library.

TODO: Implement utility functions for data manipulation, logging, and other common tasks.
"""

import pandas as pd


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
