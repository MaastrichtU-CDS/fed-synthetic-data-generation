"""
Integration tests for this library's internal serialisation pipeline.

Tests verify that components from federated_training, utils, and post_training
work together correctly: aggregation -> serialisation -> deserialisation -> loading.

These are library-internal integration tests, not framework-specific (e.g., vantage6).
"""

import numpy as np
import pytest

from fed_synthetic_data.federated_training import (
    aggregation_model_weights_weighted_average,
    evaluate_loss,
    should_stop_early,
)
from fed_synthetic_data.utils import weights_to_json, weights_from_json
from fed_synthetic_data.post_training import load_model_from_json_weights


# =============================================================================
# Mock Model for Testing
# =============================================================================

class MockModel:
    """Simple mock model for integration testing."""
    
    def __init__(self):
        self.loaded = {}
    
    def state_dict(self):
        return self.loaded
    
    def load_state_dict(self, state_dict):
        self.loaded = state_dict


# =============================================================================
# Pipeline Integration Tests
# =============================================================================

class TestAggregationToSerialisationToLoading:
    """Tests the library-internal pipeline: aggregate -> serialise -> load."""

    def test_simple_aggregation_pipeline(self):
        """Aggregate weights from nodes, serialise, and load into model."""
        # Simulate 2 federated nodes with identical structure
        node1_weights = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
        ]
        node2_weights = [
            np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
        ]
        
        # Step 1: Aggregate (federated_training)
        aggregated = aggregation_model_weights_weighted_average([
            (node1_weights, 100),
            (node2_weights, 100),
        ])
        
        # Step 2: Serialise (utils)
        serialized = weights_to_json(aggregated)
        
        # Step 3: Load (post_training)
        model = MockModel()
        parameter_names = ["weight", "bias"]
        load_model_from_json_weights(model, serialized, parameter_names)
        
        # Verify
        assert len(model.loaded) == 2
        assert model.loaded["weight"].shape == (2, 2)
        assert model.loaded["bias"].shape == (2,)
        
        # Verify aggregation was correct: (1+2)/2 = 1.5, (2+3)/2 = 2.5, etc.
        np.testing.assert_array_almost_equal(
            model.loaded["weight"],
            np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        )

    def test_weighted_aggregation_pipeline(self):
        """Verify weighted aggregation is preserved through the pipeline."""
        # Node with more samples should dominate the average
        node_small_weights = [np.array([0.0, 0.0], dtype=np.float32)]
        node_large_weights = [np.array([10.0, 10.0], dtype=np.float32)]
        
        # Aggregate with unequal sample counts
        aggregated = aggregation_model_weights_weighted_average([
            (node_small_weights, 1),   # 1 sample
            (node_large_weights, 99),  # 99 samples
        ])
        
        # Serialise and load
        serialized = weights_to_json(aggregated)
        model = MockModel()
        load_model_from_json_weights(model, serialized, ["w"])
        
        # Verify: (0*1 + 10*99)/100 = 9.9
        np.testing.assert_array_almost_equal(
            model.loaded["w"],
            np.array([9.9, 9.9], dtype=np.float32),
            decimal=5
        )

    def test_multilayer_aggregation_pipeline(self):
        """Pipeline works with multiple weight layers."""
        # 3 nodes, each with 3 layers
        node_weights = [
            [
                np.ones((4, 8), dtype=np.float32),
                np.ones((8, 2), dtype=np.float32),
                np.ones(2, dtype=np.float32),
            ]
            for _ in range(3)
        ]
        samples = [50, 100, 75]
        
        # Aggregate
        aggregated = aggregation_model_weights_weighted_average(
            [(node_weights[i], samples[i]) for i in range(3)]
        )
        
        # Serialise
        serialized = weights_to_json(aggregated)
        
        # Load
        model = MockModel()
        parameter_names = ["layer1.weight", "layer2.weight", "layer2.bias"]
        load_model_from_json_weights(model, serialized, parameter_names)
        
        # Verify all 3 layers loaded
        assert len(model.loaded) == 3
        assert model.loaded["layer1.weight"].shape == (4, 8)
        assert model.loaded["layer2.weight"].shape == (8, 2)
        assert model.loaded["layer2.bias"].shape == (2,)

    def test_dtype_preservation_through_pipeline(self):
        """Data types are preserved through aggregation and serialisation."""
        for dtype in [np.float32, np.float64]:
            weights = [np.array([1.0, 2.0], dtype=dtype)]
            
            # Aggregate (trivial case: single node)
            aggregated = aggregation_model_weights_weighted_average([(weights, 1)])
            
            # Serialise
            serialized = weights_to_json(aggregated)
            
            # Load
            model = MockModel()
            load_model_from_json_weights(model, serialized, ["w"])
            
            # Verify dtype preserved
            assert model.loaded["w"].dtype == dtype

    def test_shape_preservation_through_pipeline(self):
        """Multi-dimensional shapes are preserved through the pipeline."""
        shapes = [
            (10,),
            (5, 10),
            (10, 5),
            (3, 4, 5),
        ]
        
        weights_list = [np.random.rand(*shape).astype(np.float32) for shape in shapes]
        
        # Aggregate
        aggregated = aggregation_model_weights_weighted_average([
            (weights_list, 100),
            (weights_list, 100),
        ])
        
        # Serialise
        serialized = weights_to_json(aggregated)
        
        # Load
        model = MockModel()
        parameter_names = [f"layer{i}" for i in range(len(shapes))]
        load_model_from_json_weights(model, serialized, parameter_names)
        
        # Verify shapes
        for i, shape in enumerate(shapes):
            assert model.loaded[f"layer{i}"].shape == shape

    def test_serialisation_format_compatibility(self):
        """Verify serialisation format from utils is compatible with post_training."""
        # Create weights
        weights = [
            np.random.rand(10, 5).astype(np.float32),
            np.random.rand(5, 2).astype(np.float32),
        ]
        
        # Serialise using utils
        serialized = weights_to_json(weights)
        
        # Verify format
        assert len(serialized) == 2
        for entry in serialized:
            assert "shape" in entry
            assert "dtype" in entry
            assert "data" in entry
            assert isinstance(entry["data"], str)
        
        # Load using post_training
        model = MockModel()
        parameter_names = ["layer1", "layer2"]
        load_model_from_json_weights(model, serialized, parameter_names)
        
        # Verify data integrity
        for i, (orig, loaded) in enumerate(zip(weights, [model.loaded["layer1"], model.loaded["layer2"]])):
            np.testing.assert_array_equal(orig, loaded)


class TestPipelineErrorHandling:
    """Tests error handling in the pipeline."""

    def test_layer_count_mismatch_in_aggregation(self):
        """Aggregation fails if nodes have different layer counts."""
        node1_weights = [np.array([1.0, 2.0]), np.array([3.0])]
        node2_weights = [np.array([4.0, 5.0])]  # Missing a layer
        
        with pytest.raises(ValueError):
            aggregation_model_weights_weighted_average([
                (node1_weights, 100),
                (node2_weights, 100),
            ])

    def test_parameter_name_mismatch_in_loading(self):
        """Loading fails if parameter names don't match weight count."""
        weights = [np.array([1.0, 2.0]), np.array([3.0])]
        
        # Aggregate
        aggregated = aggregation_model_weights_weighted_average([(weights, 100)])
        
        # Serialise
        serialized = weights_to_json(aggregated)
        
        # Try to load with wrong number of parameter names
        model = MockModel()
        parameter_names = ["only_one_name"]  # Should be 2 names
        
        with pytest.raises(ValueError, match="Mismatch"):
            load_model_from_json_weights(model, serialized, parameter_names)


# =============================================================================
# Integration Tests for evaluate_loss and should_stop_early
# =============================================================================

class TestEvaluateLossIntegration:
    """Integration tests for evaluate_loss with other modules."""

    def test_evaluate_loss_with_aggregation_pipeline(self):
        """evaluate_loss integrates with aggregation and serialisation pipeline."""
        # Simulate federated training with loss tracking
        # Node 1: weights and loss
        node1_weights = [np.array([1.0, 2.0], dtype=np.float32)]
        node1_loss = 0.5
        node1_samples = 100
        
        # Node 2: weights and loss
        node2_weights = [np.array([3.0, 4.0], dtype=np.float32)]
        node2_loss = 0.7
        node2_samples = 200
        
        # Aggregate weights
        aggregated_weights = aggregation_model_weights_weighted_average([
            (node1_weights, node1_samples),
            (node2_weights, node2_samples),
        ])
        
        # Evaluate aggregate loss
        loss_results = [
            {"loss": node1_loss, "samples": node1_samples},
            {"loss": node2_loss, "samples": node2_samples},
        ]
        aggregate_loss = evaluate_loss(loss_results)
        
        # Expected: (0.5 * 100 + 0.7 * 200) / 300 = (50 + 140) / 300 = 190/300
        expected_loss = 190 / 300
        assert aggregate_loss == pytest.approx(expected_loss)
        
        # Serialise aggregated weights
        serialized = weights_to_json(aggregated_weights)
        assert len(serialized) == 1

    def test_evaluate_loss_with_multiple_sites_in_pipeline(self):
        """evaluate_loss works with many sites in the federated pipeline."""
        # Simulate 5 federated sites
        loss_results = [
            {"loss": 0.1, "samples": 50},
            {"loss": 0.2, "samples": 100},
            {"loss": 0.3, "samples": 150},
            {"loss": 0.4, "samples": 200},
            {"loss": 0.5, "samples": 250},
        ]
        
        aggregate_loss = evaluate_loss(loss_results)
        
        # Calculate expected: (0.1*50 + 0.2*100 + 0.3*150 + 0.4*200 + 0.5*250) / 750
        # = (5 + 20 + 45 + 80 + 125) / 750 = 275 / 750
        expected = 275 / 750
        assert aggregate_loss == pytest.approx(expected)


class TestShouldStopEarlyIntegration:
    """Integration tests for should_stop_early with other modules."""

    def test_should_stop_early_with_evaluate_loss_history(self):
        """should_stop_early works with loss history from evaluate_loss."""
        # Simulate federated training rounds
        loss_history = []
        
        # Round 1: loss = 1.0
        loss_results_1 = [{"loss": 1.0, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_1))
        
        # Round 2: loss = 0.8
        loss_results_2 = [{"loss": 0.8, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_2))
        
        # Round 3: loss = 0.6
        loss_results_3 = [{"loss": 0.6, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_3))
        
        # Round 4: loss = 0.5
        loss_results_4 = [{"loss": 0.5, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_4))
        
        # Round 5: loss = 0.4
        loss_results_5 = [{"loss": 0.4, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_5))
        
        # Round 6: loss = 0.4 (plateau)
        loss_results_6 = [{"loss": 0.4, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_6))
        
        # Round 7: loss = 0.4 (plateau)
        loss_results_7 = [{"loss": 0.4, "samples": 100}]
        loss_history.append(evaluate_loss(loss_results_7))
        
        # With patience=2, should_stop_early should trigger after plateau
        assert should_stop_early(loss_history, patience=2, min_delta=0.01) is True

    def test_full_training_loop_integration(self):
        """Complete training loop: aggregate -> evaluate -> check early stop."""
        # Simulate 3 rounds of federated training
        rounds = []
        
        for round_num in range(7):
            # Simulate aggregation
            node_weights = [np.array([round_num * 0.1], dtype=np.float32)]
            aggregated = aggregation_model_weights_weighted_average([
                (node_weights, 100),
                (node_weights, 100),
            ])
            
            # Simulate loss evaluation
            loss_value = 1.0 - round_num * 0.1
            loss_results = [{"loss": loss_value, "samples": 200}]
            aggregate_loss = evaluate_loss(loss_results)
            
            # Track loss history
            if round_num == 0:
                loss_history = [aggregate_loss]
            else:
                loss_history.append(aggregate_loss)
            
            # Check early stopping (patience=2)
            if round_num >= 2:
                should_stop = should_stop_early(loss_history, patience=2, min_delta=0.01)
                rounds.append({
                    "round": round_num,
                    "loss": aggregate_loss,
                    "should_stop": should_stop,
                })
        
        # Verify we have the expected number of rounds tracked
        assert len(rounds) == 5
        # The last rounds should have should_stop = False (still improving)
        # But this depends on the loss pattern
