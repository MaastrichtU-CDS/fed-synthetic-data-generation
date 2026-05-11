"""
Integration tests for post_training module.

These tests verify interactions between post_training functions and external
dependencies like PyTorch, file I/O, and other modules.
"""

import numpy as np
import pytest
import tempfile
import os

from fed_synthetic_data.utils import weights_to_json, weights_from_json


# =============================================================================
# PyTorch Integration Tests
# =============================================================================

@pytest.mark.skipif(not pytest.importorskip("torch", reason="PyTorch not available"), allow_module_level=True)
class TestPostTrainingPyTorchIntegration:
    """Integration tests requiring PyTorch."""

    def test_load_weights_into_pytorch_model(self):
        """Load numpy weights into a PyTorch model."""
        import torch
        import torch.nn as nn

        from fed_synthetic_data.post_training import load_weights_into_model

        # Create PyTorch model
        model = nn.Sequential(nn.Linear(3, 2))
        
        # Create numpy weights matching the model's state_dict
        state_dict = model.state_dict()
        weights = [v.numpy() for v in state_dict.values()]
        parameter_names = list(state_dict.keys())
        
        # Load weights
        load_weights_into_model(model, weights, parameter_names)
        
        # Verify weights match
        new_state_dict = model.state_dict()
        for key in parameter_names:
            torch.testing.assert_close(new_state_dict[key], torch.from_numpy(weights[parameter_names.index(key)]))

    def test_load_model_from_json_weights_pytorch(self):
        """Load JSON-serialised weights into a PyTorch model."""
        import torch
        import torch.nn as nn

        from fed_synthetic_data.post_training import load_model_from_json_weights

        # Create original model
        original_model = nn.Sequential(nn.Linear(3, 2))
        
        # Get weights as numpy arrays
        state_dict = original_model.state_dict()
        weights = [v.numpy() for v in state_dict.values()]
        parameter_names = list(state_dict.keys())
        
        # Serialise weights
        serialized = weights_to_json(weights)
        
        # Create new model and load
        new_model = nn.Sequential(nn.Linear(3, 2))
        load_model_from_json_weights(new_model, serialized, parameter_names)
        
        # Verify weights were loaded
        new_state_dict = new_model.state_dict()
        for key in parameter_names:
            torch.testing.assert_close(
                new_state_dict[key],
                state_dict[key]
            )

    def test_save_and_load_model_round_trip(self):
        """Save a PyTorch model and reload it."""
        import torch
        import torch.nn as nn

        from fed_synthetic_data.post_training import save_model

        # Create model with known weights
        model = nn.Sequential(nn.Linear(2, 2))
        with torch.no_grad():
            model[0].weight.fill_(1.0)
            model[0].bias.fill_(2.0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            save_model(model, path)
            
            # Load the model
            loaded_model = nn.Sequential(nn.Linear(2, 2))
            loaded_model.load_state_dict(torch.load(path))
            
            # Verify weights match
            torch.testing.assert_close(loaded_model[0].weight, model[0].weight)
            torch.testing.assert_close(loaded_model[0].bias, model[0].bias)

    def test_save_model_weights_pytorch(self):
        """Save only model weights using PyTorch."""
        import torch
        import torch.nn as nn

        from fed_synthetic_data.post_training import save_model_weights

        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pt")
            save_model_weights(model, path)
            
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            
            # Verify it's a valid PyTorch file
            loaded = torch.load(path)
            assert isinstance(loaded, dict)
            assert "weight" in loaded or list(loaded.keys())[0].endswith(".weight")


# =============================================================================
# Full Workflow Integration Tests
# =============================================================================

@pytest.mark.skipif(not pytest.importorskip("torch", reason="PyTorch not available"), allow_module_level=True)
class TestPostTrainingFullWorkflow:
    """End-to-end workflow tests: serialisation -> loading -> saving."""

    def test_complete_json_to_model_workflow(self):
        """Test full workflow: model -> serialise -> deserialize -> new model -> save."""
        import torch
        import torch.nn as nn

        from fed_synthetic_data.post_training import (
            load_model_from_json_weights,
            save_model,
        )

        # Step 1: Create and configure original model
        original_model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        with torch.no_grad():
            for param in original_model.parameters():
                param.fill_(0.5)
        
        # Step 2: Extract and serialise weights
        state_dict = original_model.state_dict()
        weights = [v.numpy() for v in state_dict.values()]
        parameter_names = list(state_dict.keys())
        serialized = weights_to_json(weights)
        
        # Step 3: Create new model and load from JSON
        new_model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        load_model_from_json_weights(new_model, serialized, parameter_names)
        
        # Step 4: Verify weights match
        new_state_dict = new_model.state_dict()
        for key in parameter_names:
            torch.testing.assert_close(new_state_dict[key], state_dict[key])
        
        # Step 5: Save the loaded model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "workflow_model.pt")
            save_model(new_model, path)
            assert os.path.exists(path)

    def test_federation_simulation_workflow(self):
        """Simulate receiving weights from federation layer and loading into model."""
        import torch
        import torch.nn as nn

        from fed_synthetic_data.federated_training import aggregation_model_weights_weighted_average
        from fed_synthetic_data.post_training import load_model_from_json_weights, save_model

        # Simulate federated training: two nodes with weights
        node1_weights = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([0.1, 0.2], dtype=np.float32),
        ]
        node2_weights = [
            np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
            np.array([0.2, 0.3], dtype=np.float32),
        ]
        
        # Aggregate weights (simulating federated aggregation)
        aggregated = aggregation_model_weights_weighted_average([
            (node1_weights, 100),
            (node2_weights, 150),
        ])
        
        # Serialise aggregated weights (as they would come from federation layer)
        serialized = weights_to_json(aggregated)
        
        # Create model and load aggregated weights
        model = nn.Sequential(nn.Linear(2, 2))
        parameter_names = ["weight", "bias"]
        load_model_from_json_weights(model, serialized, parameter_names)
        
        # Verify model has loaded the aggregated weights
        state_dict = model.state_dict()
        assert state_dict["weight"].shape == (2, 2)
        assert state_dict["bias"].shape == (2,)
        
        # Save the model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "federated_model.pt")
            save_model(model, path)
            assert os.path.exists(path)


# =============================================================================
# Cross-Module Integration Tests
# =============================================================================

class TestPostTrainingUtilsIntegration:
    """Tests verifying integration with utils.py functions."""

    def test_weights_to_json_round_trip_with_post_training(self):
        """Verify weights_to_json/weights_from_json work with post_training loading."""
        from fed_synthetic_data.post_training import load_model_from_json_weights

        class SimpleMockModel:
            def __init__(self):
                self.loaded = {}
            
            def state_dict(self):
                return self.loaded
            
            def load_state_dict(self, state_dict):
                self.loaded = state_dict

        # Original weights
        original_weights = [
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0], dtype=np.float64),
        ]
        parameter_names = ["layer1", "layer2"]
        
        # Serialise
        serialized = weights_to_json(original_weights)
        
        # Deserialise and load
        model = SimpleMockModel()
        load_model_from_json_weights(model, serialized, parameter_names)
        
        # Verify
        assert len(model.loaded) == 2
        np.testing.assert_array_equal(model.loaded["layer1"], original_weights[0])
        np.testing.assert_array_equal(model.loaded["layer2"], original_weights[1])
        assert model.loaded["layer2"].dtype == np.float64

    def test_different_dtypes_preserved_through_full_chain(self):
        """Verify dtype preservation through serialisation and loading."""
        from fed_synthetic_data.post_training import load_model_from_json_weights

        class MockModel:
            def __init__(self):
                self.data = {}
            def state_dict(self):
                return self.data
            def load_state_dict(self, sd):
                self.data = sd

        for dtype in [np.float16, np.float32, np.float64]:
            weights = [np.array([1.0, 2.0], dtype=dtype)]
            serialized = weights_to_json(weights)
            
            model = MockModel()
            load_model_from_json_weights(model, serialized, ["w"])
            
            assert model.data["w"].dtype == dtype
