"""
Utility functions for federated synthetic data generation.

This module provides general utility functions used across the library.

TODO: Implement utility functions for data manipulation, logging, and other common tasks.
"""

import pandas as pd


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the columns of a DataFrame alphabetically.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A re-indexed DataFrame with columns sorted alphabetically.
    """
    df = df.reindex(sorted(df.columns), axis=1)

    # Create a list of the column in their new order, necessary for in-flexible data generation
    column_order = df.columns.tolist()

    return df.reindex(sorted(df.columns), axis=1)
