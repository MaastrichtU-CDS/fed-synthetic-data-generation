"""
Unit tests for utils module.

This module contains unit tests for the utility functions,
including sort_columns, weights_to_json, and weights_from_json.
"""

import base64
import binascii
import json

import numpy as np
import pytest
import pandas as pd
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as np_st

from fed_synthetic_data.utils import (
    sort_columns,
    weights_from_json,
    weights_to_json,
)


class TestWeightsToJson:
    """Test cases for weights_to_json."""

    def test_single_array_round_trips(self):
        """Serialised weights can be deserialised back to the original array."""
        weights = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        result = weights_to_json(weights)
        assert len(result) == 1
        assert result[0]["dtype"] == "float32"
        assert result[0]["shape"] == (3,)
        assert isinstance(result[0]["data"], str)

    def test_multiple_layers_preserved(self):
        """All layers are serialised and the list length is preserved."""
        weights = [
            np.zeros((4, 4), dtype=np.float64),
            np.ones((4,), dtype=np.float64),
        ]
        result = weights_to_json(weights)
        assert len(result) == 2

    def test_shape_is_serialisable(self):
        """The shape field is a tuple and JSON-serialisable."""
        import json

        weights = [np.eye(3, dtype=np.float32)]
        result = weights_to_json(weights)
        # Should not raise
        json.dumps(result)

    def test_empty_list(self):
        """An empty weight list serialises to an empty list."""
        assert weights_to_json([]) == []

    def test_dtype_preserved(self):
        """The dtype string matches the original array dtype."""
        for dtype in [np.float32, np.float64]:
            weights = [np.array([1.0], dtype=dtype)]
            result = weights_to_json(weights)
            assert result[0]["dtype"] == np.dtype(dtype).str.lstrip("<>=!") or result[0][
                "dtype"
            ] == str(np.dtype(dtype))


class TestWeightsFromJson:
    """Test cases for weights_from_json."""

    def test_single_array_round_trips(self):
        """Deserialised weights match the original array values and shape."""
        original = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        recovered = weights_from_json(weights_to_json(original))

        assert len(recovered) == 1
        np.testing.assert_array_almost_equal(recovered[0], original[0])

    def test_multidimensional_shape_preserved(self):
        """2-D weight matrices are correctly restored."""
        original = [np.arange(12, dtype=np.float32).reshape(3, 4)]
        recovered = weights_from_json(weights_to_json(original))

        assert recovered[0].shape == (3, 4)
        np.testing.assert_array_equal(recovered[0], original[0])

    def test_multiple_layers_round_trip(self):
        """All layers survive a full serialise -> deserialise round-trip."""
        original = [
            np.random.rand(8, 8).astype(np.float32),
            np.zeros(8, dtype=np.float32),
            np.random.rand(8, 4).astype(np.float32),
            np.zeros(4, dtype=np.float32),
        ]
        recovered = weights_from_json(weights_to_json(original))

        assert len(recovered) == len(original)
        for orig, rec in zip(original, recovered):
            np.testing.assert_array_almost_equal(rec, orig)

    def test_empty_list(self):
        """An empty entry list deserialises to an empty list."""
        assert weights_from_json([]) == []

    def test_dtype_preserved_after_round_trip(self):
        """The dtype of the recovered array matches the original."""
        for dtype in [np.float32, np.float64]:
            original = [np.array([1.0, 2.0], dtype=dtype)]
            recovered = weights_from_json(weights_to_json(original))
            assert recovered[0].dtype == np.dtype(dtype)


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

    def test_sort_columns_duplicate_columns(self):
        """Test sorting DataFrame with duplicate column names."""
        # Create DataFrame with duplicate columns
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "a"])

        # Sort should handle duplicates - pandas keeps both
        result = sort_columns(df)

        # Should have 3 columns, sorted alphabetically
        assert len(result.columns) == 3
        # Both 'a' columns should be present
        assert list(result.columns).count("a") == 2

    def test_sort_columns_single_row(self):
        """Test sorting DataFrame with only one row."""
        df = pd.DataFrame({"z": [1], "a": [2], "m": [3]})

        result = sort_columns(df)

        assert list(result.columns) == ["a", "m", "z"]
        assert len(result) == 1
        assert result.loc[0, "a"] == 2
        assert result.loc[0, "m"] == 3
        assert result.loc[0, "z"] == 1

    def test_sort_columns_single_cell(self):
        """Test sorting DataFrame with only one row and one column."""
        df = pd.DataFrame({"a": [1]})

        result = sort_columns(df)

        assert list(result.columns) == ["a"]
        assert len(result) == 1
        assert result.loc[0, "a"] == 1

    def test_sort_columns_all_same_column_names(self):
        """Test sorting DataFrame where all columns have the same name."""
        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "a"])

        result = sort_columns(df)

        # All columns should still be there
        assert len(result.columns) == 3
        assert all(c == "a" for c in result.columns)

    def test_sort_columns_mixed_types(self):
        """Test sorting DataFrame with mixed column name types."""
        df = pd.DataFrame({1: [1, 2], "a": [3, 4], 2: [5, 6]})

        result = sort_columns(df)

        # Should sort with numbers first (1, 2), then strings ("a")
        assert list(result.columns) == [1, 2, "a"]

    def test_sort_columns_empty_with_explicit_order(self):
        """Test sorting empty DataFrame with explicit column order."""
        df = pd.DataFrame()

        result = sort_columns(df, column_order=["a", "b", "c"])

        # Empty DataFrame should remain empty
        assert result.empty
        assert list(result.columns) == ["a", "b", "c"]


class TestWeightsFromJsonMalformed:
    """Tests for weights_from_json with invalid/malformed inputs."""

    def test_missing_data_key(self):
        """Missing 'data' key raises KeyError."""
        entries = [{"shape": (2,), "dtype": "float32"}]
        with pytest.raises(KeyError):
            weights_from_json(entries)

    def test_missing_shape_key(self):
        """Missing 'shape' key raises KeyError."""
        entries = [
            {"data": base64.b64encode(np.array([1.0, 2.0]).tobytes()).decode(), "dtype": "float32"}
        ]
        with pytest.raises(KeyError):
            weights_from_json(entries)

    def test_missing_dtype_key(self):
        """Missing 'dtype' key raises KeyError."""
        entries = [
            {"data": base64.b64encode(np.array([1.0, 2.0]).tobytes()).decode(), "shape": (2,)}
        ]
        with pytest.raises(KeyError):
            weights_from_json(entries)

    def test_invalid_base64_data(self):
        """Invalid base64 string raises error."""
        entries = [{"data": "not-valid-base64!!!", "shape": (2,), "dtype": "float32"}]
        with pytest.raises((binascii.Error, ValueError)):
            weights_from_json(entries)

    def test_invalid_dtype_string(self):
        """Unrecognised dtype string raises TypeError."""
        data_str = base64.b64encode(np.array([1.0, 2.0]).tobytes()).decode()
        entries = [{"data": data_str, "shape": (2,), "dtype": "invalid_dtype_xyz"}]
        with pytest.raises(TypeError):
            weights_from_json(entries)

    def test_shape_mismatch_with_data_length(self):
        """Shape tuple doesn't match actual data length."""
        data = base64.b64encode(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).tobytes()).decode()
        entries = [{"data": data, "shape": (2,), "dtype": "float32"}]  # 4 elements vs shape (2,)
        with pytest.raises(ValueError):
            weights_from_json(entries)


class TestWeightsRoundTripHypothesis:
    """Property-based tests for serialisation round-trips."""

    @given(
        n_layers=st.integers(min_value=0, max_value=10),
        array_shapes=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=20), st.integers(min_value=0, max_value=20)
            ),
            min_size=0,
            max_size=10,
        ),
    )
    def test_round_trip_preserves_shape(self, n_layers, array_shapes):
        """Round-trip preserves array shapes for arbitrary layer configurations."""
        np.random.seed(42)
        original = [np.random.rand(*shape).astype(np.float32) for shape in array_shapes[:n_layers]]
        recovered = weights_from_json(weights_to_json(original))
        assert len(recovered) == len(original)
        for orig, rec in zip(original, recovered):
            assert rec.shape == orig.shape

    @given(
        arrays=st.lists(
            np_st.arrays(
                dtype=np.float32,
                shape=st.tuples(
                    st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=10)
                ),
                elements=st.floats(
                    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
                ),
            ),
            min_size=0,
            max_size=5,
        )
    )
    def test_round_trip_preserves_values(self, arrays):
        """Round-trip preserves exact float32 values."""
        recovered = weights_from_json(weights_to_json(arrays))
        assert len(recovered) == len(arrays)
        for orig, rec in zip(arrays, recovered):
            np.testing.assert_array_equal(rec, orig)

    @given(dtype=st.sampled_from([np.float16, np.float32, np.float64, np.int32, np.int64]))
    def test_round_trip_preserves_dtype(self, dtype):
        """Round-trip preserves the original dtype."""
        original = [np.array([1.0, 2.0, 3.0], dtype=dtype)]
        recovered = weights_from_json(weights_to_json(original))
        assert recovered[0].dtype == np.dtype(dtype)


class TestJsonPayloadSchema:
    """Tests to pin the JSON payload schema contract."""

    def test_weights_to_json_schema_compliance(self):
        """Every entry has exactly shape, dtype, data keys with correct types."""
        weights = [np.random.rand(3, 4).astype(np.float32), np.random.rand(10).astype(np.float64)]
        result = weights_to_json(weights)

        for entry in result:
            assert set(entry.keys()) == {"shape", "dtype", "data"}
            assert isinstance(entry["shape"], tuple)
            assert all(isinstance(d, int) for d in entry["shape"])
            assert isinstance(entry["dtype"], str)
            assert isinstance(entry["data"], str)
            # Verify it's valid base64
            base64.b64decode(entry["data"], validate=True)

        # Entire structure must be JSON-serialisable
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

    def test_schema_is_stable_across_dtypes(self):
        """Schema is identical regardless of input dtype."""
        for dtype in [np.float16, np.float32, np.float64]:
            w = np.array([[1.0, 2.0]], dtype=dtype)
            entry = weights_to_json([w])[0]
            assert set(entry.keys()) == {"shape", "dtype", "data"}
            assert entry["shape"] == (1, 2)
            assert entry["dtype"] == np.dtype(dtype).str
