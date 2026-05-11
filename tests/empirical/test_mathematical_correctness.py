"""
Empirical tests for mathematical correctness of library functions.

These tests verify that the aggregation and serialisation functions
produce mathematically correct results.
"""

import numpy as np
import pytest

from fed_synthetic_data.federated_training import aggregation_model_weights_weighted_average
from fed_synthetic_data.utils import weights_to_json, weights_from_json


class TestAggregationMathematics:
    """Verify aggregation functions produce mathematically correct results."""

    def test_simple_weighted_average_two_nodes(self):
        """Verify weighted average with two nodes of equal size."""
        # Node 1: weights [1.0, 2.0]
        # Node 2: weights [3.0, 4.0]
        # Equal samples: (1+3)/2 = 2.0, (2+4)/2 = 3.0
        w1 = [np.array([1.0, 2.0], dtype=np.float32)]
        w2 = [np.array([3.0, 4.0], dtype=np.float32)]
        
        result = aggregation_model_weights_weighted_average([
            (w1, 100),
            (w2, 100),
        ])
        
        np.testing.assert_array_almost_equal(
            result[0],
            np.array([2.0, 3.0], dtype=np.float32)
        )

    def test_weighted_average_unequal_samples(self):
        """Verify weighted average with unequal sample counts."""
        # Node 1: weights [1.0, 2.0] with 1 sample
        # Node 2: weights [3.0, 4.0] with 3 samples
        # Weighted avg: (1*1 + 3*3)/4 = 10/4 = 2.5
        #              (2*1 + 4*3)/4 = 14/4 = 3.5
        w1 = [np.array([1.0, 2.0], dtype=np.float32)]
        w2 = [np.array([3.0, 4.0], dtype=np.float32)]
        
        result = aggregation_model_weights_weighted_average([
            (w1, 1),
            (w2, 3),
        ])
        
        np.testing.assert_array_almost_equal(
            result[0],
            np.array([2.5, 3.5], dtype=np.float32)
        )

    def test_multilayer_weighted_average(self):
        """Verify weighted average across multiple weight layers."""
        # Node 1: layer1=[1.0], layer2=[2.0] with 2 samples
        # Node 2: layer1=[3.0], layer2=[4.0] with 8 samples
        # Layer 1: (1*2 + 3*8)/10 = 26/10 = 2.6
        # Layer 2: (2*2 + 4*8)/10 = 36/10 = 3.6
        w1 = [np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32)]
        w2 = [np.array([3.0], dtype=np.float32), np.array([4.0], dtype=np.float32)]
        
        result = aggregation_model_weights_weighted_average([
            (w1, 2),
            (w2, 8),
        ])
        
        np.testing.assert_array_almost_equal(result[0], np.array([2.6], dtype=np.float32))
        np.testing.assert_array_almost_equal(result[1], np.array([3.6], dtype=np.float32))

    def test_single_node_returns_unchanged(self):
        """Single node's weights should be returned unchanged."""
        w = [np.array([5.0, 6.0, 7.0], dtype=np.float32)]
        
        result = aggregation_model_weights_weighted_average([(w, 42)])
        
        np.testing.assert_array_almost_equal(result[0], w[0])

    def test_zero_weight_node_ignored(self):
        """Node with zero samples should not affect the average."""
        w1 = [np.array([10.0, 20.0], dtype=np.float32)]
        w2 = [np.array([0.0, 0.0], dtype=np.float32)]  # Should be ignored
        
        result = aggregation_model_weights_weighted_average([
            (w1, 100),
            (w2, 0),
        ])
        
        # Result should equal w1 since w2 has 0 samples
        np.testing.assert_array_almost_equal(result[0], w1[0])

    def test_multidimensional_weight_matrices(self):
        """Verify weighted average works with multi-dimensional arrays."""
        # 2x2 weight matrices
        # Node 1: [[1, 2], [3, 4]] with 1 sample
        # Node 2: [[5, 6], [7, 8]] with 1 sample
        # Average: [[3, 4], [5, 6]]
        w1 = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
        w2 = [np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)]
        
        result = aggregation_model_weights_weighted_average([
            (w1, 1),
            (w2, 1),
        ])
        
        expected = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result[0], expected)

    def test_different_dtypes_preserved_in_aggregation(self):
        """Verify aggregation preserves data types."""
        for dtype in [np.float32, np.float64]:
            w1 = [np.array([1.0, 2.0], dtype=dtype)]
            w2 = [np.array([3.0, 4.0], dtype=dtype)]
            
            result = aggregation_model_weights_weighted_average([
                (w1, 1),
                (w2, 1),
            ])
            
            assert result[0].dtype == dtype

    def test_multiple_nodes_with_various_weights(self):
        """Verify aggregation with many nodes of varying sizes."""
        # 5 nodes with different weights and sample counts
        weights_and_samples = [
            ([np.array([1.0], dtype=np.float32)], 10),
            ([np.array([2.0], dtype=np.float32)], 20),
            ([np.array([3.0], dtype=np.float32)], 30),
            ([np.array([4.0], dtype=np.float32)], 40),
            ([np.array([5.0], dtype=np.float32)], 50),
        ]
        
        # Expected: (1*10 + 2*20 + 3*30 + 4*40 + 5*50) / 150
        # = (10 + 40 + 90 + 160 + 250) / 150 = 550 / 150 = 3.666...
        result = aggregation_model_weights_weighted_average(weights_and_samples)
        
        expected = 550.0 / 150.0
        np.testing.assert_almost_equal(result[0][0], expected, decimal=5)


class TestSerialisationFidelity:
    """Verify serialisation preserves numerical properties."""

    def test_round_trip_preserves_exact_values(self):
        """Serialisation and deserialisation preserve exact values."""
        original = [
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            np.array([-1.5, 0.0, 1.5], dtype=np.float64),
        ]
        
        serialized = weights_to_json(original)
        recovered = weights_from_json(serialized)
        
        for orig, rec in zip(original, recovered):
            np.testing.assert_array_equal(orig, rec)

    def test_round_trip_preserves_dtype(self):
        """Serialisation preserves data types."""
        for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]:
            original = [np.array([1, 2, 3], dtype=dtype)]
            recovered = weights_from_json(weights_to_json(original))
            assert recovered[0].dtype == dtype

    def test_round_trip_preserves_shape(self):
        """Serialisation preserves array shapes."""
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

    def test_serialisation_of_empty_list(self):
        """Empty weight list can be serialised and deserialised."""
        original = []
        serialized = weights_to_json(original)
        recovered = weights_from_json(serialized)
        assert recovered == []

    def test_serialisation_of_single_weight(self):
        """Single weight array serialisation."""
        original = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        recovered = weights_from_json(weights_to_json(original))
        np.testing.assert_array_equal(original[0], recovered[0])

    def test_serialisation_preserves_negative_values(self):
        """Negative values are preserved through serialisation."""
        original = [np.array([-5.0, -3.0, 0.0, 3.0, 5.0], dtype=np.float32)]
        recovered = weights_from_json(weights_to_json(original))
        np.testing.assert_array_equal(original[0], recovered[0])

    def test_serialisation_preserves_extreme_values(self):
        """Extreme values are preserved through serialisation."""
        original = [
            np.array([np.finfo(np.float32).min, np.finfo(np.float32).max], dtype=np.float32),
        ]
        recovered = weights_from_json(weights_to_json(original))
        # Use almost_equal due to potential precision issues with extreme values
        np.testing.assert_array_almost_equal(original[0], recovered[0])

    def test_serialisation_format_consistency(self):
        """Serialised format is consistent and complete."""
        weights = [np.array([1.0, 2.0], dtype=np.float32)]
        serialized = weights_to_json(weights)
        
        assert len(serialized) == 1
        entry = serialized[0]
        
        # Check all required fields are present
        assert "shape" in entry
        assert "dtype" in entry
        assert "data" in entry
        
        # Check field types
        assert isinstance(entry["shape"], tuple)
        assert isinstance(entry["dtype"], str)
        assert isinstance(entry["data"], str)
        
        # Check shape is correct
        assert entry["shape"] == (2,)
        assert entry["dtype"] == "float32"


@pytest.mark.skipif(not pytest.importorskip("torch", reason="PyTorch not available"), 
                    allow_module_level=True)
class TestPyTorchTensorCompatibility:
    """Verify our functions work with PyTorch tensors (as returned by fed-mostlyai-engine)."""

    def test_weights_to_json_with_pytorch_tensors(self):
        """PyTorch tensors can be serialised directly without numpy conversion."""
        import torch

        # Create PyTorch tensors (as fed-mostlyai-engine returns them)
        pytorch_weights = [
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([4.0, 5.0], dtype=torch.float64),
        ]

        # This should work without converting to numpy first
        serialized = weights_to_json(pytorch_weights)

        assert len(serialized) == 2
        assert serialized[0]["shape"] == (3,)
        assert serialized[0]["dtype"] == "float32"
        assert serialized[1]["shape"] == (2,)
        assert serialized[1]["dtype"] == "float64"
        assert isinstance(serialized[0]["data"], str)

    def test_weights_from_json_with_pytorch_serialized_tensors(self):
        """PyTorch tensors can be serialised and deserialised to numpy arrays."""
        import torch

        # Create PyTorch tensors
        pytorch_weights = [
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
            torch.tensor([4.0, 5.0], dtype=torch.float64),
        ]

        # Serialise and deserialize
        serialized = weights_to_json(pytorch_weights)
        recovered = weights_from_json(serialized)

        # Should return numpy arrays
        assert len(recovered) == 2
        assert isinstance(recovered[0], np.ndarray)
        assert isinstance(recovered[1], np.ndarray)
        
        # Values should match
        np.testing.assert_array_almost_equal(
            recovered[0],
            pytorch_weights[0].numpy()
        )
        np.testing.assert_array_almost_equal(
            recovered[1],
            pytorch_weights[1].numpy()
        )

    def test_aggregation_with_pytorch_tensors(self):
        """Aggregation works with PyTorch tensors as input."""
        import torch

        # Create PyTorch tensors for two nodes
        node1_weights = [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([0.1], dtype=torch.float32),
        ]
        node2_weights = [
            torch.tensor([3.0, 4.0], dtype=torch.float32),
            torch.tensor([0.2], dtype=torch.float32),
        ]

        # Aggregate (should work with PyTorch tensors)
        result = aggregation_model_weights_weighted_average([
            (node1_weights, 100),
            (node2_weights, 100),
        ])

        # Result should be numpy arrays (from the aggregation function)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        
        # Values should be correct
        np.testing.assert_array_almost_equal(
            result[0],
            np.array([2.0, 3.0], dtype=np.float32)
        )
        np.testing.assert_array_almost_equal(
            result[1],
            np.array([0.15], dtype=np.float32)
        )

    def test_full_pytorch_tensor_workflow(self):
        """Complete workflow: PyTorch tensors -> aggregate -> serialise -> deserialize."""
        import torch

        # Simulate two federated nodes with PyTorch tensors
        node1_weights = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            torch.tensor([0.1, 0.2], dtype=torch.float32),
        ]
        node2_weights = [
            torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
            torch.tensor([0.2, 0.3], dtype=torch.float32),
        ]

        # Step 1: Aggregate (with PyTorch tensors directly)
        aggregated = aggregation_model_weights_weighted_average([
            (node1_weights, 100),
            (node2_weights, 100),
        ])

        # Step 2: Serialise
        serialized = weights_to_json(aggregated)

        # Step 3: Deserialize
        recovered = weights_from_json(serialized)

        # Verify
        assert len(recovered) == 2
        assert isinstance(recovered[0], np.ndarray)
        assert isinstance(recovered[1], np.ndarray)
        assert recovered[0].shape == (2, 2)
        assert recovered[1].shape == (2,)
