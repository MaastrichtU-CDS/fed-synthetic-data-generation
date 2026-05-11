"""
Integration tests for utils.py functions with other library components.

Tests verify that sort_columns and serialisation utilities work correctly
within the library's workflows.
"""

import numpy as np
import pytest
import pandas as pd

from fed_synthetic_data.utils import sort_columns, weights_to_json, weights_from_json

# =============================================================================
# sort_columns Integration Tests
# =============================================================================


class TestSortColumnsFederatedWorkflow:
    """Tests for sort_columns in federated synthetic data workflows."""

    def test_consistent_column_order_across_nodes(self):
        """All federated nodes produce DataFrames with the same column order."""
        # Node 1: columns in one order
        df_node1 = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [25, 30, 35],
                "diagnosis": ["A", "B", "C"],
                "treatment": ["X", "Y", "Z"],
            }
        )

        # Node 2: columns in different order
        df_node2 = pd.DataFrame(
            {
                "treatment": ["P", "Q"],
                "patient_id": [4, 5],
                "diagnosis": ["D", "E"],
                "age": [28, 32],
            }
        )

        # Node 3: columns in yet another order
        df_node3 = pd.DataFrame(
            {
                "age": [22, 24],
                "treatment": ["R", "S"],
                "patient_id": [6, 7],
                "diagnosis": ["F", "G"],
            }
        )

        # Derive canonical order from first node
        canonical_order = list(sort_columns(df_node1).columns)

        # Apply to all nodes
        df_node1_sorted = sort_columns(df_node1, column_order=canonical_order)
        df_node2_sorted = sort_columns(df_node2, column_order=canonical_order)
        df_node3_sorted = sort_columns(df_node3, column_order=canonical_order)

        # Verify all have the same column order
        assert list(df_node1_sorted.columns) == canonical_order
        assert list(df_node2_sorted.columns) == canonical_order
        assert list(df_node3_sorted.columns) == canonical_order

    def test_derive_canonical_order_from_central_node(self):
        """Central node derives order, edge nodes apply it - typical pattern."""
        # Central node (e.g., server) defines the canonical order
        central_df = pd.DataFrame(
            {"z_score": [1.0, 2.0], "a_value": [10, 20], "m_metric": [100, 200]}
        )

        # Central node derives the order
        canonical_order = list(sort_columns(central_df).columns)

        # Edge node receives data in arbitrary order
        edge_df = pd.DataFrame({"m_metric": [300, 400], "a_value": [30, 40], "z_score": [3.0, 4.0]})

        # Edge node applies the canonical order
        edge_df_sorted = sort_columns(edge_df, column_order=canonical_order)

        assert list(edge_df_sorted.columns) == canonical_order
        assert list(edge_df_sorted.columns) == list(sort_columns(central_df).columns)

    def test_column_order_with_missing_columns(self):
        """Handling missing columns in edge nodes."""
        # Canonical order includes all possible columns
        canonical_order = ["id", "feature_a", "feature_b", "feature_c", "label"]

        # Edge node has only a subset of columns
        edge_df = pd.DataFrame({"feature_b": [1, 2, 3], "id": [10, 20, 30], "label": [0, 1, 0]})

        # Apply canonical order
        sorted_df = sort_columns(edge_df, column_order=canonical_order)

        # reindex produces all columns in canonical order; missing ones are NaN
        assert list(sorted_df.columns) == canonical_order
        assert pd.isna(sorted_df["feature_a"]).all()
        assert pd.isna(sorted_df["feature_c"]).all()

    def test_column_order_with_extra_columns(self):
        """Handling extra columns in edge nodes."""
        # Canonical order
        canonical_order = ["id", "feature_a", "label"]

        # Edge node has extra column
        edge_df = pd.DataFrame(
            {"id": [1, 2], "feature_a": [10, 20], "extra_col": [100, 200], "label": [0, 1]}
        )

        # Apply canonical order
        sorted_df = sort_columns(edge_df, column_order=canonical_order)

        # reindex selects only columns in canonical_order; extra_col is dropped
        assert list(sorted_df.columns) == canonical_order
        assert "extra_col" not in sorted_df.columns

    def test_alphabetical_sort_as_fallback(self):
        """When no column_order provided, sorts alphabetically."""
        df_unsorted = pd.DataFrame({"zebra": [1, 2], "apple": [3, 4], "banana": [5, 6]})

        sorted_df = sort_columns(df_unsorted)

        assert list(sorted_df.columns) == ["apple", "banana", "zebra"]

    def test_numeric_column_names(self):
        """Sorting works with numeric column names."""
        df = pd.DataFrame({3: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]})

        sorted_df = sort_columns(df)

        assert list(sorted_df.columns) == [1, 2, 3]

    def test_preserves_data_values(self):
        """Sorting only reorders columns, doesn't modify data."""
        df = pd.DataFrame({"c": [1, 2, 3], "a": [4, 5, 6], "b": [7, 8, 9]})

        sorted_df = sort_columns(df)

        # Data should be unchanged
        assert sorted_df["a"].tolist() == [4, 5, 6]
        assert sorted_df["b"].tolist() == [7, 8, 9]
        assert sorted_df["c"].tolist() == [1, 2, 3]

    def test_preserves_index(self):
        """Sorting preserves DataFrame index."""
        df = pd.DataFrame({"z": [1, 2], "a": [3, 4]}, index=["row1", "row2"])

        sorted_df = sort_columns(df)

        assert list(sorted_df.index) == ["row1", "row2"]
        assert sorted_df.loc["row1", "a"] == 3
        assert sorted_df.loc["row2", "z"] == 2


# =============================================================================
# Serialisation Utilities Integration Tests
# =============================================================================


class TestSerialisationUtilitiesIntegration:
    """Tests for weights_to_json and weights_from_json integration."""

    def test_round_trip_preserves_all_properties(self):
        """Round-trip through serialisation preserves all weight properties."""
        original_weights = [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float16),
        ]

        # Serialise
        serialized = weights_to_json(original_weights)

        # Deserialise
        recovered = weights_from_json(serialized)

        # Verify
        assert len(recovered) == len(original_weights)
        for orig, rec in zip(original_weights, recovered):
            np.testing.assert_array_equal(orig, rec)
            assert orig.dtype == rec.dtype
            assert orig.shape == rec.shape

    def test_empty_list_round_trip(self):
        """Empty weight lists can be serialised and deserialised."""
        assert weights_from_json(weights_to_json([])) == []

    def test_single_weight_round_trip(self):
        """Single weight array round-trip."""
        original = [np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)]
        recovered = weights_from_json(weights_to_json(original))

        np.testing.assert_array_equal(original[0], recovered[0])
        assert original[0].dtype == recovered[0].dtype

    def test_serialized_format_is_json_compatible(self):
        """Serialized weights can be serialised to JSON string."""
        import json

        weights = [np.array([1.0, 2.0], dtype=np.float32)]
        serialized = weights_to_json(weights)

        # Should not raise
        json_str = json.dumps(serialized)

        # And can be loaded back
        loaded = json.loads(json_str)
        recovered = weights_from_json(loaded)

        np.testing.assert_array_equal(weights[0], recovered[0])

    def test_different_dtypes_preserved(self):
        """Different numpy dtypes are preserved through serialisation."""
        dtypes = [np.float16, np.float32, np.float64, np.int32, np.int64]

        for dtype in dtypes:
            original = [np.array([1, 2, 3], dtype=dtype)]
            serialized = weights_to_json(original)
            recovered = weights_from_json(serialized)

            assert recovered[0].dtype == dtype

    def test_multidimensional_shapes_preserved(self):
        """Multi-dimensional array shapes are preserved."""
        shapes = [
            (10,),
            (5, 10),
            (2, 3, 4),
            (2, 2, 2, 2),
        ]

        for shape in shapes:
            original = [np.random.rand(*shape).astype(np.float32)]
            recovered = weights_from_json(weights_to_json(original))

            assert recovered[0].shape == shape
