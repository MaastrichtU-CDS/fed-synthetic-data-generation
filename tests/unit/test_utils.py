"""
Unit tests for utils module.

This module contains unit tests for the utility functions,
including sort_columns.
"""

import pytest
import pandas as pd

from fed_synthetic_data.utils import sort_columns


class TestSortColumns:
    """Test cases for the sort_columns function."""

    def test_sort_columns_basic(self):
        """Test sorting columns in basic DataFrame."""
        # Create DataFrame with unsorted columns
        df = pd.DataFrame({"zebra": [1, 2, 3], "apple": [4, 5, 6], "banana": [7, 8, 9]})

        # Call sort_columns
        result = sort_columns(df)

        # Verify columns are sorted alphabetically
        expected_columns = ["apple", "banana", "zebra"]
        assert list(result.columns) == expected_columns

        # Verify data is preserved
        assert result["apple"].tolist() == [4, 5, 6]
        assert result["banana"].tolist() == [7, 8, 9]
        assert result["zebra"].tolist() == [1, 2, 3]

    def test_sort_columns_already_sorted(self):
        """Test sorting columns that are already sorted."""
        # Create DataFrame with already sorted columns
        df = pd.DataFrame({"apple": [1, 2, 3], "banana": [4, 5, 6], "zebra": [7, 8, 9]})

        # Call sort_columns
        result = sort_columns(df)

        # Verify columns remain sorted
        expected_columns = ["apple", "banana", "zebra"]
        assert list(result.columns) == expected_columns

    def test_sort_columns_single_column(self):
        """Test sorting DataFrame with single column."""
        # Create DataFrame with single column
        df = pd.DataFrame({"zebra": [1, 2, 3]})

        # Call sort_columns
        result = sort_columns(df)

        # Verify single column is preserved
        assert list(result.columns) == ["zebra"]
        assert result["zebra"].tolist() == [1, 2, 3]

    def test_sort_columns_empty_dataframe(self):
        """Test sorting empty DataFrame."""
        # Create empty DataFrame
        df = pd.DataFrame()

        # Call sort_columns
        result = sort_columns(df)

        # Verify empty DataFrame is returned
        assert result.empty
        assert list(result.columns) == []

    def test_sort_columns_numeric_columns(self):
        """Test sorting DataFrame with numeric column names."""
        # Create DataFrame with numeric column names
        df = pd.DataFrame({3: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})

        # Call sort_columns
        result = sort_columns(df)

        # Verify columns are sorted numerically
        expected_columns = [1, 2, 3]
        assert list(result.columns) == expected_columns

    def test_sort_columns_preserves_index(self):
        """Test that sort_columns preserves the DataFrame index."""
        # Create DataFrame with custom index
        df = pd.DataFrame({"zebra": [1, 2, 3], "apple": [4, 5, 6]}, index=["row1", "row2", "row3"])

        # Call sort_columns
        result = sort_columns(df)

        # Verify index is preserved
        assert list(result.index) == ["row1", "row2", "row3"]
        assert result["apple"]["row1"] == 4
        assert result["zebra"]["row3"] == 3

    def test_sort_columns_returns_new_dataframe(self):
        """Test that sort_columns returns a new DataFrame, not modifying the original."""
        # Create DataFrame
        df = pd.DataFrame({"zebra": [1, 2, 3], "apple": [4, 5, 6]})

        original_columns = list(df.columns)

        # Call sort_columns
        result = sort_columns(df)

        # Verify original DataFrame is unchanged
        assert list(df.columns) == original_columns

        # Verify result is different
        assert list(result.columns) != original_columns
        assert list(result.columns) == ["apple", "zebra"]
