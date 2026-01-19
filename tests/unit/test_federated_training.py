"""
Unit tests for federated training functionality.
"""

import pytest
import numpy as np
from fed_synthetic_data.federated_training import (
    FederatedTrainer,
    aggregate_model_updates,
    compute_node_weight,
    secure_aggregation,
)


@pytest.mark.unit
class TestFederatedTrainer:
    """Test cases for FederatedTrainer class."""

    def test_initialization(self):
        """Test that trainer initializes with correct parameters."""
        trainer = FederatedTrainer(
            aggregation_method="fedavg",
            num_rounds=10
        )
        
        assert trainer.aggregation_method == "fedavg"
        assert trainer.num_rounds == 10
        assert trainer.current_round == 0
        assert len(trainer.nodes) == 0

    def test_initialization_defaults(self):
        """Test that trainer initializes with default parameters."""
        trainer = FederatedTrainer()
        
        assert trainer.aggregation_method == "fedavg"
        assert trainer.num_rounds == 10
        assert trainer.current_round == 0

    def test_add_node(self):
        """Test adding nodes to the trainer."""
        trainer = FederatedTrainer()
        
        trainer.add_node("node_1", {"data": "test_data_1"})
        trainer.add_node("node_2", {"data": "test_data_2"})
        
        assert len(trainer.nodes) == 2
        assert trainer.nodes[0]["id"] == "node_1"
        assert trainer.nodes[1]["id"] == "node_2"

    def test_train_round(self):
        """Test executing a single training round."""
        trainer = FederatedTrainer(num_rounds=5)
        
        initial_round = trainer.current_round
        result = trainer.train_round()
        
        assert trainer.current_round == initial_round + 1
        assert isinstance(result, dict)
        assert "round" in result
        assert "status" in result

    def test_train_full_process(self):
        """Test executing the full training process."""
        trainer = FederatedTrainer(num_rounds=3)
        
        result = trainer.train()
        
        assert isinstance(result, dict)
        assert result["rounds_completed"] == 3
        assert len(result["results"]) == 3
        assert trainer.current_round == 3

    def test_get_global_model(self):
        """Test retrieving the global model."""
        trainer = FederatedTrainer()
        model = trainer.get_global_model()
        
        # Should return None initially
        assert model is None


@pytest.mark.unit
class TestAggregateModelUpdates:
    """Test cases for aggregate_model_updates function."""

    def test_basic_aggregation(self):
        """Test basic model aggregation."""
        local_models = [
            {"param1": 1.0, "param2": 2.0},
            {"param1": 3.0, "param2": 4.0},
        ]
        
        result = aggregate_model_updates(local_models)
        assert isinstance(result, dict)

    def test_aggregation_with_weights(self):
        """Test aggregation with custom weights."""
        local_models = [
            {"param1": 1.0},
            {"param1": 3.0},
        ]
        weights = [0.3, 0.7]
        
        result = aggregate_model_updates(local_models, weights=weights)
        assert isinstance(result, dict)

    def test_aggregation_raises_on_empty_models(self):
        """Test that aggregation raises error with no models."""
        with pytest.raises(ValueError, match="No local models"):
            aggregate_model_updates([])

    def test_aggregation_raises_on_weight_mismatch(self):
        """Test that aggregation raises error when weights don't match models."""
        local_models = [{"param1": 1.0}, {"param1": 2.0}]
        weights = [0.5]  # Only one weight for two models
        
        with pytest.raises(ValueError, match="must match"):
            aggregate_model_updates(local_models, weights=weights)

    def test_aggregation_with_different_methods(self):
        """Test aggregation with different methods."""
        local_models = [{"param1": 1.0}, {"param1": 2.0}]
        
        # FedAvg
        result_fedavg = aggregate_model_updates(local_models, aggregation_method="fedavg")
        assert isinstance(result_fedavg, dict)
        
        # FedProx
        result_fedprox = aggregate_model_updates(local_models, aggregation_method="fedprox")
        assert isinstance(result_fedprox, dict)

    def test_aggregation_raises_on_unknown_method(self):
        """Test that unknown aggregation method raises error."""
        local_models = [{"param1": 1.0}]
        
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_model_updates(local_models, aggregation_method="unknown")


@pytest.mark.unit
class TestComputeNodeWeight:
    """Test cases for compute_node_weight function."""

    def test_basic_weight_computation(self):
        """Test basic node weight computation."""
        weight = compute_node_weight(100, 1000)
        assert weight == 0.1

    def test_equal_data_sizes(self):
        """Test weight computation with equal sizes."""
        weight = compute_node_weight(500, 500)
        assert weight == 1.0

    def test_raises_on_zero_total(self):
        """Test that zero total size raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_node_weight(100, 0)

    def test_raises_on_negative_total(self):
        """Test that negative total size raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_node_weight(100, -100)


@pytest.mark.unit
class TestSecureAggregation:
    """Test cases for secure_aggregation function."""

    def test_basic_secure_aggregation(self):
        """Test basic secure aggregation."""
        updates = [
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ]
        
        result = secure_aggregation(updates)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [2.5, 3.5, 4.5])

    def test_single_update(self):
        """Test secure aggregation with single update."""
        updates = [np.array([1.0, 2.0, 3.0])]
        
        result = secure_aggregation(updates)
        
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_raises_on_empty_updates(self):
        """Test that empty updates raise error."""
        with pytest.raises(ValueError, match="No local updates"):
            secure_aggregation([])

    def test_with_encryption_method(self):
        """Test secure aggregation with encryption method."""
        updates = [
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
        ]
        
        result = secure_aggregation(updates, encryption_method="homomorphic")
        assert isinstance(result, np.ndarray)
