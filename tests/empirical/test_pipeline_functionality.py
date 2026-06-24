"""
Empirical tests for pipeline functionality.

These tests verify that the complete pipeline of this library's components
produces functional, usable results.
"""

import importlib.util
import os
import pytest
import tempfile

import numpy as np
import pandas as pd

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

from fed_synthetic_data.federated_training import aggregation_model_weights_weighted_average
from fed_synthetic_data.utils import weights_to_json, weights_from_json, sort_columns
from fed_synthetic_data.post_training import (
    load_weights_into_model,
    load_model_from_json_weights,
    save_model,
)

# =============================================================================
# Mock Model for Testing
# =============================================================================


class MockModel:
    """Mock model that implements the minimal interface for testing."""

    def __init__(self):
        self.state_dict_data = {}
        self.loaded_state_dict = None
        self.last_inference_input = None
        self.last_inference_output = None

    def state_dict(self):
        return self.state_dict_data

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict
        self.state_dict_data.update(state_dict)

    def infer(self, x):
        """Mock inference - just returns sum of weights * input."""
        # Simple inference: output = sum of all weights * input
        if not self.loaded_state_dict:
            raise ValueError("Model has no loaded weights")

        weight_sum = sum(np.sum(np.asarray(w)) for w in self.loaded_state_dict.values())
        self.last_inference_input = x
        self.last_inference_output = weight_sum * x
        return self.last_inference_output


# =============================================================================
# Pipeline Functionality Tests
# =============================================================================


class TestAggregationToLoadingPipeline:
    """Test the complete pipeline: aggregation -> serialisation -> loading."""

    def test_pipeline_produces_loadable_model(self):
        """Aggregated weights can be serialised and loaded into a model."""
        # Simulate 3 federated nodes
        node1_weights = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.1], dtype=np.float32),
        ]
        node2_weights = [
            np.array([3.0, 4.0], dtype=np.float32),
            np.array([0.2], dtype=np.float32),
        ]
        node3_weights = [
            np.array([5.0, 6.0], dtype=np.float32),
            np.array([0.3], dtype=np.float32),
        ]

        # Step 1: Aggregate
        aggregated = aggregation_model_weights_weighted_average(
            [
                (node1_weights, 100),
                (node2_weights, 200),
                (node3_weights, 150),
            ]
        )

        # Step 2: Serialise
        serialized = weights_to_json(aggregated)

        # Step 3: Load into model
        model = MockModel()
        parameter_names = ["weights", "bias"]
        load_model_from_json_weights(model, serialized, parameter_names)

        # Verify model has loaded weights
        assert model.loaded_state_dict is not None
        assert "weights" in model.loaded_state_dict
        assert "bias" in model.loaded_state_dict
        assert model.loaded_state_dict["weights"].shape == (2,)
        assert model.loaded_state_dict["bias"].shape == (1,)

    def test_pipeline_preserves_weight_properties(self):
        """Pipeline preserves dtype and shape through aggregation and loading."""
        # Create weights with specific dtype
        dtype = np.float32
        node_weights = [
            np.random.rand(10, 5).astype(dtype),
            np.random.rand(5, 2).astype(dtype),
        ]

        # Aggregate
        aggregated = aggregation_model_weights_weighted_average(
            [
                (node_weights, 100),
                (node_weights, 100),
            ]
        )

        # Serialise and load
        serialized = weights_to_json(aggregated)
        model = MockModel()
        parameter_names = ["layer1", "layer2"]
        load_model_from_json_weights(model, serialized, parameter_names)

        # Verify properties preserved
        assert np.asarray(model.loaded_state_dict["layer1"]).dtype == dtype
        assert np.asarray(model.loaded_state_dict["layer2"]).dtype == dtype
        assert model.loaded_state_dict["layer1"].shape == (10, 5)
        assert model.loaded_state_dict["layer2"].shape == (5, 2)

    def test_pipeline_with_multiple_nodes_and_layers(self):
        """Pipeline works with many nodes and multiple weight layers."""
        # 5 nodes, each with 4 layers
        num_nodes = 5
        num_layers = 4
        layer_shapes = [(10,), (10, 5), (5, 5), (5, 2)]

        node_weights = []
        for _ in range(num_nodes):
            weights = [np.random.rand(*shape).astype(np.float32) for shape in layer_shapes]
            node_weights.append(weights)

        samples_per_node = [100, 150, 200, 250, 300]

        # Aggregate
        aggregated = aggregation_model_weights_weighted_average(
            [(node_weights[i], samples_per_node[i]) for i in range(num_nodes)]
        )

        # Serialise and load
        serialized = weights_to_json(aggregated)
        model = MockModel()
        parameter_names = [f"layer{i}" for i in range(num_layers)]
        load_model_from_json_weights(model, serialized, parameter_names)

        # Verify all layers loaded correctly
        assert len(model.loaded_state_dict) == num_layers
        for i, shape in enumerate(layer_shapes):
            assert model.loaded_state_dict[f"layer{i}"].shape == shape

    def test_pipeline_handles_edge_cases(self):
        """Pipeline handles edge cases correctly."""
        # Single node, single layer
        single_weights = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        aggregated = aggregation_model_weights_weighted_average([(single_weights, 1)])
        serialized = weights_to_json(aggregated)
        model = MockModel()
        load_model_from_json_weights(model, serialized, ["single_layer"])

        assert "single_layer" in model.loaded_state_dict
        np.testing.assert_array_equal(model.loaded_state_dict["single_layer"], single_weights[0])


class TestLoadedModelFunctionality:
    """Test that loaded models are functional."""

    def test_loaded_model_can_perform_inference(self):
        """Model loaded via pipeline can perform inference."""
        # Create and aggregate weights
        node_weights = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([0.5], dtype=np.float32),
        ]

        aggregated = aggregation_model_weights_weighted_average(
            [
                (node_weights, 100),
            ]
        )

        # Load into model
        model = MockModel()
        parameter_names = ["weights", "bias"]
        serialized = weights_to_json(aggregated)
        load_model_from_json_weights(model, serialized, parameter_names)

        # Verify model can perform inference
        test_input = 2.0
        result = model.infer(test_input)

        assert model.last_inference_input == test_input
        assert model.last_inference_output is not None
        assert isinstance(result, (int, float, np.number))

    def test_loaded_model_produces_consistent_output(self):
        """Loaded model produces consistent output for same input."""
        node_weights = [
            np.array([1.0, 2.0], dtype=np.float32),
        ]

        aggregated = aggregation_model_weights_weighted_average([(node_weights, 100)])
        serialized = weights_to_json(aggregated)

        model = MockModel()
        load_model_from_json_weights(model, serialized, ["weights"])

        # Run inference twice with same input
        input1 = 3.0
        input2 = 3.0

        output1 = model.infer(input1)
        output2 = model.infer(input2)

        assert output1 == output2

    def test_different_weights_produce_different_outputs(self):
        """Models with different weights produce different outputs."""
        # Create two different sets of weights
        weights_set1 = [np.array([1.0, 2.0], dtype=np.float32)]
        weights_set2 = [np.array([3.0, 4.0], dtype=np.float32)]

        # Load into two models
        model1 = MockModel()
        model2 = MockModel()

        serialized1 = weights_to_json(weights_set1)
        serialized2 = weights_to_json(weights_set2)

        load_model_from_json_weights(model1, serialized1, ["weights"])
        load_model_from_json_weights(model2, serialized2, ["weights"])

        # Same input should produce different outputs
        test_input = 2.0
        output1 = model1.infer(test_input)
        output2 = model2.infer(test_input)

        assert output1 != output2


class TestSaveAndReloadPipeline:
    """Test saving and reloading models in the pipeline."""

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
    class TestPyTorchSaveReload:
        """PyTorch-specific save/reload tests."""

        def test_save_and_reload_preserves_weights(self):
            """Save and reload preserves model weights."""
            import torch
            import torch.nn as nn

            # Create a model with known weights
            model = nn.Sequential(nn.Linear(3, 2))
            with torch.no_grad():
                model[0].weight.fill_(1.0)
                model[0].bias.fill_(2.0)

            # Save the model
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test_model.pt")
                save_model(model, path)

                # Reload into a new model
                new_model = nn.Sequential(nn.Linear(3, 2))
                new_model.load_state_dict(torch.load(path))

                # Verify weights match
                torch.testing.assert_close(new_model[0].weight, model[0].weight)
                torch.testing.assert_close(new_model[0].bias, model[0].bias)

        def test_save_and_reload_preserves_inference(self):
            """Reloaded model produces same inference results."""
            import torch
            import torch.nn as nn

            # Create model
            torch.manual_seed(42)
            model = nn.Sequential(nn.Linear(4, 2))

            # Save
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test_model.pt")
                save_model(model, path)

                # Reload
                new_model = nn.Sequential(nn.Linear(4, 2))
                new_model.load_state_dict(torch.load(path))

                # Test inference
                test_input = torch.randn(5, 4)
                with torch.no_grad():
                    output1 = model(test_input)
                    output2 = new_model(test_input)

                torch.testing.assert_close(output1, output2)

        def test_full_pipeline_with_save(self):
            """Complete pipeline including save: aggregate -> serialise -> load -> save -> reload."""
            import torch
            import torch.nn as nn

            # Create weights from multiple "nodes"
            torch.manual_seed(42)
            model1 = nn.Linear(3, 2)
            model2 = nn.Linear(3, 2)

            # Get weights as numpy
            weights1 = [p.detach().numpy() for p in model1.parameters()]
            weights2 = [p.detach().numpy() for p in model2.parameters()]

            # Aggregate
            aggregated = aggregation_model_weights_weighted_average(
                [
                    (weights1, 100),
                    (weights2, 100),
                ]
            )

            # Serialise and load into new PyTorch model
            serialized = weights_to_json(aggregated)
            new_model = nn.Linear(3, 2)
            parameter_names = ["weight", "bias"]
            load_model_from_json_weights(new_model, serialized, parameter_names)

            # Save the loaded model
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "pipeline_model.pt")
                save_model(new_model, path)

                # Reload
                final_model = nn.Linear(3, 2)
                final_model.load_state_dict(torch.load(path))

                # Verify weights match
                torch.testing.assert_close(
                    final_model.weight, torch.from_numpy(new_model.weight.detach().numpy())
                )
                torch.testing.assert_close(
                    final_model.bias, torch.from_numpy(new_model.bias.detach().numpy())
                )


class TestFedMostlyAIEngineWorkflow:
    """Tests simulating fed-mostlyai-engine usage patterns."""

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
    class TestStateDictWorkflow:
        """Tests for workflow with PyTorch state_dict (as from fed-mostlyai-engine)."""

        def test_full_workflow_with_pytorch_state_dict(self):
            """Complete workflow using PyTorch state_dict format (fed-mostlyai-engine pattern)."""
            import torch
            import torch.nn as nn

            # Simulate fed-mostlyai-engine: create model and get state_dict
            model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
            torch.manual_seed(42)

            # Get state_dict (this is what fed-mostlyai-engine returns in federated_state["model_weights"])
            state_dict = model.state_dict()

            # Convert to list format for our aggregation function
            # In practice, each federated node would provide its own state_dict
            node1_weights = [v.detach().cpu().numpy() for v in state_dict.values()]
            node2_weights = [v.detach().cpu().numpy() for v in state_dict.values()]
            parameter_names = list(state_dict.keys())

            # Step 1: Aggregate (simulating federated aggregation)
            aggregated = aggregation_model_weights_weighted_average(
                [
                    (node1_weights, 100),
                    (node2_weights, 150),
                ]
            )

            # Step 2: Serialise
            serialized = weights_to_json(aggregated)

            # Step 3: Load into new model
            new_model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
            load_model_from_json_weights(new_model, serialized, parameter_names)

            # Step 4: Verify model can perform inference
            test_input = torch.randn(5, 4)
            with torch.no_grad():
                output = new_model(test_input)

            assert output.shape == (5, 2)

        def test_state_dict_to_aggregation_to_loaded_model(self):
            """Verify state_dict weights can be aggregated and loaded back."""
            import torch
            import torch.nn as nn

            # Create reference model
            torch.manual_seed(123)
            reference_model = nn.Sequential(nn.Linear(3, 2))

            # Get reference state_dict
            reference_state_dict = reference_model.state_dict()

            # Simulate two nodes with the same architecture
            # (In federated learning, nodes might have slightly different weights)
            torch.manual_seed(456)
            node1_model = nn.Sequential(nn.Linear(3, 2))
            torch.manual_seed(789)
            node2_model = nn.Sequential(nn.Linear(3, 2))

            # Extract weights from both nodes
            node1_weights = [v.detach().cpu().numpy() for v in node1_model.state_dict().values()]
            node2_weights = [v.detach().cpu().numpy() for v in node2_model.state_dict().values()]
            parameter_names = list(reference_state_dict.keys())

            # Aggregate
            aggregated = aggregation_model_weights_weighted_average(
                [
                    (node1_weights, 100),
                    (node2_weights, 100),
                ]
            )

            # Serialise and load
            serialized = weights_to_json(aggregated)
            loaded_model = nn.Sequential(nn.Linear(3, 2))
            load_model_from_json_weights(loaded_model, serialized, parameter_names)

            # Verify loaded model has the correct shape state_dict
            loaded_state_dict = loaded_model.state_dict()
            assert len(loaded_state_dict) == len(reference_state_dict)
            for key in reference_state_dict:
                assert loaded_state_dict[key].shape == reference_state_dict[key].shape

        def test_federated_state_compatibility(self):
            """Test compatibility with fed-mostlyai-engine's federated_state format."""
            import torch
            import torch.nn as nn

            # Simulate what fed-mostlyai-engine returns in federated_state
            model = nn.Sequential(nn.Linear(5, 3))
            torch.manual_seed(42)

            # This mimics: federated_state["model_weights"] from fed-mostlyai-engine
            model_weights = model.state_dict()

            # Convert to our list format (in practice, this would be done per-node)
            # For a single node, just wrap in a list
            weights_list = [model_weights[key].detach().cpu().numpy() for key in model_weights]
            parameter_names = list(model_weights.keys())

            # Now use our pipeline
            # Aggregate (trivial case: single node)
            aggregated = aggregation_model_weights_weighted_average(
                [
                    (weights_list, 100),
                ]
            )

            # Serialise
            serialized = weights_to_json(aggregated)

            # Load into new model
            new_model = nn.Sequential(nn.Linear(5, 3))
            load_model_from_json_weights(new_model, serialized, parameter_names)

            # Verify the new model has the same weights
            new_state_dict = new_model.state_dict()
            for key in model_weights:
                torch.testing.assert_close(new_state_dict[key], model_weights[key])


# =============================================================================
# Realistic Federated Scenario Tests
# =============================================================================


class TestRealisticFederatedScenario:
    """Tests using weight magnitudes and architectures representative of real federated learning."""

    def test_aggregation_with_xavier_initialised_weights(self):
        """Aggregation of Xavier-initialised weights stays within the expected magnitude range."""
        # Xavier uniform init for a 128->64 layer: limit = sqrt(6 / (128 + 64)) ≈ 0.177
        rng = np.random.default_rng(42)
        fan_in, fan_out = 128, 64
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        num_nodes = 5
        samples_per_node = [1000, 1500, 800, 2000, 1200]

        node_weights = [
            [
                rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32),
                np.zeros(fan_out, dtype=np.float32),
            ]
            for _ in range(num_nodes)
        ]

        aggregated = aggregation_model_weights_weighted_average(
            list(zip(node_weights, samples_per_node))
        )

        assert aggregated[0].shape == (fan_out, fan_in)
        assert np.all(np.abs(aggregated[0]) <= limit + 1e-6)
        # Biases are all zero across nodes, so the average must also be zero
        np.testing.assert_array_almost_equal(aggregated[1], np.zeros(fan_out, dtype=np.float32))

    def test_aggregation_with_multilayer_realistic_architecture(self):
        """Aggregation preserves structure and dtype across a typical multi-layer architecture."""
        # Typical generator-like architecture: 128 -> 64 -> 32 -> 16
        layer_configs = [(128, 64), (64, 32), (32, 16)]
        rng = np.random.default_rng(0)

        def _xavier_layer(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return [
                rng.uniform(-limit, limit, size=(fan_out, fan_in)).astype(np.float32),
                np.zeros(fan_out, dtype=np.float32),
            ]

        num_nodes = 4
        samples_per_node = [2000, 3000, 1500, 2500]

        node_weights = []
        for _ in range(num_nodes):
            layers = []
            for fan_in, fan_out in layer_configs:
                layers.extend(_xavier_layer(fan_in, fan_out))
            node_weights.append(layers)

        aggregated = aggregation_model_weights_weighted_average(
            list(zip(node_weights, samples_per_node))
        )

        # Weights and biases alternate; verify shape and dtype for every layer
        weight_layers = aggregated[0::2]
        bias_layers = aggregated[1::2]

        for agg_w, (fan_in, fan_out) in zip(weight_layers, layer_configs):
            assert agg_w.shape == (fan_out, fan_in)
            assert agg_w.dtype == np.float32

        for agg_b, (_, fan_out) in zip(bias_layers, layer_configs):
            assert agg_b.shape == (fan_out,)
            assert agg_b.dtype == np.float32

    def test_aggregation_dominated_by_largest_node(self):
        """Aggregated result is strongly pulled towards the node with by far the most samples."""
        rng = np.random.default_rng(7)
        shape = (64, 32)

        dominant_weights = rng.normal(0.0, 0.1, size=shape).astype(np.float32)
        small_weights = [rng.normal(0.0, 0.1, size=shape).astype(np.float32) for _ in range(4)]

        node_weights_and_samples = [([dominant_weights], 10_000)] + [
            ([w], 10) for w in small_weights
        ]

        aggregated = aggregation_model_weights_weighted_average(node_weights_and_samples)

        # With 10000 vs 4×10 samples the result must be very close to the dominant node
        np.testing.assert_array_almost_equal(aggregated[0], dominant_weights, decimal=2)

    def test_aggregation_convergence_over_multiple_rounds(self):
        """Aggregating near-converged nodes repeatedly stays close to the shared mean."""
        rng = np.random.default_rng(99)
        shape = (32, 16)
        num_nodes = 6
        num_rounds = 3
        samples = [1000] * num_nodes

        # True mean represents a converged global model
        true_mean = rng.normal(0.0, 0.05, size=shape).astype(np.float32)

        aggregated = None
        for _ in range(num_rounds):
            # Each node's weights are the true mean plus small local noise
            node_weights = [
                [true_mean + rng.normal(0.0, 0.01, size=shape).astype(np.float32)]
                for _ in range(num_nodes)
            ]
            aggregated = aggregation_model_weights_weighted_average(
                list(zip(node_weights, samples))
            )

        np.testing.assert_array_almost_equal(aggregated[0], true_mean, decimal=1)

    def test_aggregation_with_heterogeneous_realistic_sample_counts(self):
        """Weighted average is numerically correct with realistic, heterogeneous sample counts."""
        # Simulate nodes representing hospitals/sites of different sizes
        rng = np.random.default_rng(13)
        shape = (16,)
        samples_per_node = [500, 1200, 3000, 250, 800, 4500]
        total = sum(samples_per_node)

        node_weights = [
            rng.normal(0.0, 0.1, size=shape).astype(np.float32) for _ in samples_per_node
        ]

        aggregated = aggregation_model_weights_weighted_average(
            [([w], n) for w, n in zip(node_weights, samples_per_node)]
        )

        # Manually compute the expected weighted average
        expected = sum(w * n for w, n in zip(node_weights, samples_per_node)) / total
        np.testing.assert_array_almost_equal(aggregated[0], expected.astype(np.float32), decimal=5)

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
    class TestRealisticPyTorchWorkflow:
        """Realistic end-to-end tests using actual PyTorch weight initialisations."""

        def test_aggregation_of_freshly_initialised_models(self):
            """Freshly initialised PyTorch models aggregate to weights of the correct shape and magnitude."""
            import torch
            import torch.nn as nn

            def _make_model():
                return nn.Sequential(
                    nn.Linear(64, 128),
                    nn.Linear(128, 64),
                    nn.Linear(64, 32),
                )

            num_nodes = 5
            samples_per_node = [1000, 2000, 1500, 800, 3000]

            node_weights = []
            for seed in range(num_nodes):
                torch.manual_seed(seed)
                model = _make_model()
                node_weights.append([p.detach().cpu().numpy() for p in model.parameters()])

            aggregated = aggregation_model_weights_weighted_average(
                list(zip(node_weights, samples_per_node))
            )

            expected_shapes = [(128, 64), (128,), (64, 128), (64,), (32, 64), (32,)]
            for agg_w, expected_shape in zip(aggregated, expected_shapes):
                assert agg_w.shape == expected_shape
                assert agg_w.dtype == np.float32

            # Aggregated weight matrices should stay within typical Kaiming/Xavier range
            for agg_w in aggregated[0::2]:
                assert np.abs(agg_w).max() < 1.0, "Aggregated weights are unexpectedly large"

        def test_aggregated_model_produces_finite_inference(self):
            """A model loaded with realistic aggregated weights produces finite (non-NaN) inference."""
            import torch
            import torch.nn as nn

            def _make_model():
                return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))

            num_nodes = 4
            samples_per_node = [500, 750, 1000, 250]
            parameter_names = ["0.weight", "0.bias", "2.weight", "2.bias"]

            node_weights = []
            for seed in range(num_nodes):
                torch.manual_seed(seed * 10)
                model = _make_model()
                node_weights.append([p.detach().cpu().numpy() for p in model.parameters()])

            aggregated = aggregation_model_weights_weighted_average(
                list(zip(node_weights, samples_per_node))
            )

            serialized = weights_to_json(aggregated)
            new_model = _make_model()
            load_model_from_json_weights(new_model, serialized, parameter_names)

            torch.manual_seed(0)
            test_input = torch.randn(32, 16)
            with torch.no_grad():
                output = new_model(test_input)

            assert output.shape == (32, 8)
            assert torch.isfinite(output).all(), "Inference produced NaN or Inf values"


# =============================================================================
# Empirical Tests for sort_columns
# =============================================================================


class TestSortColumnsEmpirical:
    """Empirical tests for sort_columns in realistic scenarios."""

    def test_large_dataframe_performance(self):
        """sort_columns performs well on large DataFrames."""
        import time

        # Create a large DataFrame with many columns in random order
        np.random.seed(42)
        num_rows = 10000
        num_cols = 100

        columns = [f"col_{i:04d}" for i in np.random.permutation(range(num_cols))]
        df = pd.DataFrame(np.random.randn(num_rows, num_cols), columns=columns)

        # Time the sorting
        start_time = time.time()
        result = sort_columns(df)
        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 1 second for 10k rows, 100 cols)
        assert elapsed < 1.0

        # Verify columns are sorted
        assert list(result.columns) == sorted(result.columns)

        # Verify data integrity (first and last row should match)
        np.testing.assert_array_equal(result.iloc[0].values, df.iloc[0][result.columns].values)
        np.testing.assert_array_equal(result.iloc[-1].values, df.iloc[-1][result.columns].values)

    def test_many_duplicate_columns(self):
        """sort_columns handles DataFrames with many duplicate column names."""
        # Create DataFrame with many duplicate columns
        df = pd.DataFrame(
            np.random.rand(100, 50), columns=["feature"] * 50  # All columns named "feature"
        )

        result = sort_columns(df)

        # All columns should still be present
        assert len(result.columns) == 50
        assert all(c == "feature" for c in result.columns)

        # Data should be preserved
        assert result.shape == df.shape

    def test_mixed_numeric_string_columns(self):
        """sort_columns handles mixed numeric and string column names correctly."""
        df = pd.DataFrame(
            {
                10: [1, 2, 3],
                "b": [4, 5, 6],
                1: [7, 8, 9],
                "a": [10, 11, 12],
                5: [13, 14, 15],
            }
        )

        result = sort_columns(df)

        # Numeric columns should come first (sorted), then string columns (sorted)
        expected_columns = [1, 5, 10, "a", "b"]
        assert list(result.columns) == expected_columns

    def test_very_long_column_names(self):
        """sort_columns handles very long column names."""
        long_names = [
            "a" * 100,
            "b" * 100,
            "c" * 100,
        ]
        df = pd.DataFrame(np.random.rand(10, 3), columns=long_names)

        result = sort_columns(df)

        assert list(result.columns) == sorted(long_names)

    def test_unicode_column_names(self):
        """sort_columns handles unicode column names."""
        df = pd.DataFrame(
            {
                "ζ": [1, 2, 3],
                "α": [4, 5, 6],
                "β": [7, 8, 9],
            }
        )

        result = sort_columns(df)

        # Unicode should sort correctly
        assert list(result.columns) == sorted(["ζ", "α", "β"])

    def test_column_order_consistency_across_multiple_calls(self):
        """sort_columns produces consistent results across multiple calls."""
        df = pd.DataFrame(
            {
                "zebra": [1, 2],
                "apple": [3, 4],
                "banana": [5, 6],
            }
        )

        # Call sort_columns multiple times
        results = [sort_columns(df) for _ in range(10)]

        # All results should be identical
        for i in range(1, 10):
            pd.testing.assert_frame_equal(results[0], results[i])

    def test_federated_column_consistency(self):
        """sort_columns ensures consistency across federated nodes with different schemas."""
        # Node 1 has columns in one order
        node1_data = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [25, 30, 35],
                "diagnosis": ["A", "B", "C"],
            }
        )

        # Node 2 has same columns but different order
        node2_data = pd.DataFrame(
            {
                "diagnosis": ["D", "E"],
                "patient_id": [4, 5],
                "age": [28, 32],
            }
        )

        # Both nodes sort their columns
        node1_sorted = sort_columns(node1_data)
        node2_sorted = sort_columns(node2_data)

        # Column order should be identical
        assert list(node1_sorted.columns) == list(node2_sorted.columns)

        # Now test with explicit column order from node1
        node2_explicit = sort_columns(node2_data, column_order=list(node1_sorted.columns))
        assert list(node2_explicit.columns) == list(node1_sorted.columns)


# =============================================================================
# Pipeline Error Handling Tests
# =============================================================================


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""

    def test_length_mismatch_in_loading(self):
        """Loading fails gracefully when parameter names don't match weights."""
        weights = [np.array([1.0, 2.0]), np.array([3.0])]

        aggregated = aggregation_model_weights_weighted_average([(weights, 100)])
        serialized = weights_to_json(aggregated)

        model = MockModel()
        parameter_names = ["only_one"]  # Should be 2 names

        with pytest.raises(ValueError, match="Mismatch"):
            load_model_from_json_weights(model, serialized, parameter_names)

    def test_layer_count_mismatch_in_aggregation(self):
        """Aggregation fails gracefully when nodes have different layer counts."""
        node1_weights = [np.array([1.0]), np.array([2.0])]
        node2_weights = [np.array([3.0])]  # Missing a layer

        with pytest.raises(ValueError):
            aggregation_model_weights_weighted_average(
                [
                    (node1_weights, 100),
                    (node2_weights, 100),
                ]
            )

    def test_empty_pipeline(self):
        """Empty inputs are handled correctly throughout the pipeline."""
        # Empty aggregation
        aggregated = aggregation_model_weights_weighted_average([])
        assert aggregated == []

        # Empty serialisation
        serialized = weights_to_json([])
        assert serialized == []

        # Empty loading
        model = MockModel()
        result = load_model_from_json_weights(model, [], [])
        assert result is model
        assert model.loaded_state_dict == {}
