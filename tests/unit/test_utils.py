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

    def test_sort_columns_explicit_order(self):
        """Test reordering columns using an explicit column_order list."""
        df = pd.DataFrame({"apple": [1, 2, 3], "banana": [4, 5, 6], "zebra": [7, 8, 9]})

        result = sort_columns(df, column_order=["zebra", "apple", "banana"])

        assert list(result.columns) == ["zebra", "apple", "banana"]
        assert result["zebra"].tolist() == [7, 8, 9]
        assert result["apple"].tolist() == [1, 2, 3]
        assert result["banana"].tolist() == [4, 5, 6]

    def test_sort_columns_explicit_order_matches_alphabetical(self):
        """Test that passing the sorted column list explicitly gives the same result as no argument."""
        df = pd.DataFrame({"zebra": [1, 2, 3], "apple": [4, 5, 6], "banana": [7, 8, 9]})

        auto_sorted = sort_columns(df)
        explicit = sort_columns(df, column_order=list(auto_sorted.columns))

        assert list(explicit.columns) == list(auto_sorted.columns)

    def test_sort_columns_explicit_order_is_federated_workflow(self):
        """Test the federated use-case: derive order on one node, apply it on another."""
        df_node1 = pd.DataFrame({"zebra": [1, 2], "apple": [3, 4], "banana": [5, 6]})
        df_node2 = pd.DataFrame({"banana": [7, 8], "zebra": [9, 10], "apple": [11, 12]})

        # Node 1 derives the canonical order
        df_node1_sorted = sort_columns(df_node1)
        canonical_order = df_node1_sorted.columns.tolist()

        # Node 2 applies the same order
        df_node2_sorted = sort_columns(df_node2, column_order=canonical_order)

        assert list(df_node1_sorted.columns) == list(df_node2_sorted.columns)

    def test_sort_columns_explicit_order_subset_introduces_nans(self):
        """Test that a column_order omitting a column produces NaN for the missing column."""
        df = pd.DataFrame({"apple": [1, 2], "banana": [3, 4], "zebra": [5, 6]})

        result = sort_columns(df, column_order=["apple", "banana"])

        assert list(result.columns) == ["apple", "banana"]
        assert "zebra" not in result.columns

    def test_sort_columns_explicit_order_extra_column_introduces_nans(self):
        """Test that a column_order referencing a non-existent column produces a NaN column."""
        import math

        df = pd.DataFrame({"apple": [1, 2], "banana": [3, 4]})

        result = sort_columns(df, column_order=["apple", "banana", "zebra"])

        assert list(result.columns) == ["apple", "banana", "zebra"]
        assert all(math.isnan(v) for v in result["zebra"].tolist())
