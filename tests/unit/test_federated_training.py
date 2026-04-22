"""
Unit tests for federated_training module.

This module contains unit tests for the serialisation helpers and the
weighted-average aggregation function used in federated synthetic data training.
"""

import numpy as np
import pytest

from fed_synthetic_data.federated_training import (
    aggregation_model_weights_weighted_average,
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
            assert result[0]["dtype"] == np.dtype(dtype).str.lstrip("<>=!") or result[0]["dtype"] == str(np.dtype(dtype))


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
        """All layers survive a full serialise → deserialise round-trip."""
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


class TestAggregationModelWeightsWeightedAverage:
    """Test cases for aggregation_model_weights_weighted_average."""

    def test_equal_weights_equal_samples(self):
        """Equal weights from equal-sized nodes should produce the same weights."""
        w = [np.array([1.0, 2.0, 3.0])]
        results = [(w, 10), (w, 10)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated[0], w[0])

    def test_larger_node_dominates(self):
        """A node with more samples should pull the average towards its weights."""
        w_small = [np.array([0.0, 0.0])]
        w_large = [np.array([10.0, 10.0])]
        results = [(w_small, 1), (w_large, 99)]
        aggregated = aggregation_model_weights_weighted_average(results)

        # Result should be close to w_large
        assert aggregated[0][0] > 9.0

    def test_two_nodes_known_result(self):
        """Manually verify weighted average for a simple two-node case."""
        w1 = [np.array([2.0, 4.0])]
        w2 = [np.array([4.0, 8.0])]
        # 1 sample vs 3 samples → expected: (2*1 + 4*3)/4=3.5, (4*1 + 8*3)/4=7.0
        results = [(w1, 1), (w2, 3)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated[0], np.array([3.5, 7.0]))

    def test_multiple_layers(self):
        """Aggregation works correctly across multiple weight layers."""
        w1 = [np.array([1.0]), np.array([2.0])]
        w2 = [np.array([3.0]), np.array([4.0])]
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated[0], np.array([2.0]))
        np.testing.assert_array_almost_equal(aggregated[1], np.array([3.0]))

    def test_single_node(self):
        """A single node's weights should be returned unchanged."""
        w = [np.array([5.0, 6.0, 7.0])]
        results = [(w, 42)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated[0], w[0])

    def test_layer_count_mismatch_raises(self):
        """Mismatched layer counts between nodes should raise an error."""
        w1 = [np.array([1.0]), np.array([2.0])]
        w2 = [np.array([1.0])]  # one layer fewer
        results = [(w1, 10), (w2, 10)]

        with pytest.raises(ValueError):
            aggregation_model_weights_weighted_average(results)