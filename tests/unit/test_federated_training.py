"""
Unit tests for federated_training module.

This module contains unit tests for the weighted-average aggregation function,
evaluate_loss, and should_stop_early used in federated synthetic data training.
"""

import math
import pytest

import numpy as np

from fed_synthetic_data.federated_training import (
    aggregation_model_weights_weighted_average,
    evaluate_loss,
    lr_determination,
    reducelr_on_plateau_step,
    should_stop_early,
)


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

    def test_empty_results_list(self):
        """Empty results list returns empty list."""
        result = aggregation_model_weights_weighted_average([])
        assert result == []

    def test_negative_weights(self):
        """Negative weight values are handled correctly."""
        w1 = [np.array([-1.0, -2.0], dtype=np.float32)]
        w2 = [np.array([-3.0, -4.0], dtype=np.float32)]
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(
            aggregated[0], np.array([-2.0, -3.0], dtype=np.float32)
        )

    def test_very_large_weights(self):
        """Very large weight values don't cause overflow."""
        w1 = [np.array([1e30, 1e30], dtype=np.float64)]
        w2 = [np.array([2e30, 2e30], dtype=np.float64)]
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)

        # Use relative tolerance for large values where absolute tolerance fails
        np.testing.assert_allclose(
            aggregated[0], np.array([1.5e30, 1.5e30], dtype=np.float64), rtol=1e-10
        )

    def test_very_small_weights(self):
        """Very small weight values are handled correctly."""
        w1 = [np.array([1e-30, 1e-30], dtype=np.float64)]
        w2 = [np.array([2e-30, 2e-30], dtype=np.float64)]
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(
            aggregated[0], np.array([1.5e-30, 1.5e-30], dtype=np.float64)
        )

    def test_mixed_positive_negative_weights(self):
        """Mixed positive and negative weights average correctly."""
        w1 = [np.array([-1.0, 2.0], dtype=np.float32)]
        w2 = [np.array([3.0, -4.0], dtype=np.float32)]
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated[0], np.array([1.0, -1.0], dtype=np.float32))


class TestEvaluateLoss:
    """Test cases for evaluate_loss."""

    def test_single_site(self):
        """Single site loss is returned as-is."""
        loss_results = [{"loss": 0.5, "samples": 100}]
        assert evaluate_loss(loss_results) == 0.5

    def test_two_sites_equal_samples(self):
        """Two sites with equal samples: simple average."""
        loss_results = [{"loss": 0.4, "samples": 100}, {"loss": 0.6, "samples": 100}]
        assert evaluate_loss(loss_results) == 0.5

    def test_two_sites_unequal_samples(self):
        """Loss is weighted by sample count."""
        loss_results = [{"loss": 0.2, "samples": 10}, {"loss": 0.8, "samples": 90}]
        # (0.2 * 10 + 0.8 * 90) / 100 = 0.74
        assert evaluate_loss(loss_results) == pytest.approx(0.74)

    def test_empty_list_raises(self):
        """Empty loss_results should raise ValueError."""
        with pytest.raises(ValueError, match="loss_results cannot be empty"):
            evaluate_loss([])

    def test_zero_samples_raises(self):
        """Zero total samples should raise ValueError."""
        loss_results = [{"loss": 0.5, "samples": 0}]
        with pytest.raises(ValueError, match="total number of samples must be greater than zero"):
            evaluate_loss(loss_results)

    def test_multiple_sites_weighted_average(self):
        """Verify weighted average across multiple sites."""
        loss_results = [
            {"loss": 1.0, "samples": 10},
            {"loss": 2.0, "samples": 20},
            {"loss": 3.0, "samples": 30},
        ]
        # (1*10 + 2*20 + 3*30) / 60 = (10 + 40 + 90) / 60 = 140/60 ≈ 2.3333
        assert evaluate_loss(loss_results) == pytest.approx(140 / 60)


class TestShouldStopEarly:
    """Test cases for should_stop_early."""

    def test_not_enough_history(self):
        """Returns False when history length <= patience."""
        loss_history = [0.5, 0.4, 0.3]
        assert should_stop_early(loss_history, patience=5) is False

    def test_exact_patience_length(self):
        """Returns False when history length equals patience."""
        loss_history = [0.5, 0.4, 0.3, 0.2, 0.1]
        assert should_stop_early(loss_history, patience=5) is False

    def test_improving_loss(self):
        """Returns False when loss is still improving."""
        loss_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        assert should_stop_early(loss_history, patience=5) is False

    def test_not_improving_within_delta(self):
        """Returns True when an improvement is within min_delta (plateau counts as no improvement)."""
        # Loss plateaus at ~0.1, which is within min_delta of previous best (0.1)
        # Plateau counts as no improvement, so should stop
        loss_history = [0.5, 0.4, 0.3, 0.2, 0.1, 0.1 + 1e-5, 0.1 + 2e-5]
        assert should_stop_early(loss_history, patience=2, min_delta=1e-4) is True

    def test_not_improving_beyond_delta(self):
        """Returns True when loss has not improved by min_delta for patience iterations."""
        loss_history = [0.5, 0.4, 0.3, 0.2, 0.1, 0.1 + 1e-3, 0.1 + 2e-3]
        assert should_stop_early(loss_history, patience=2, min_delta=1e-4) is True

    def test_constant_loss(self):
        """Returns True when loss is constant for patience iterations."""
        loss_history = [0.5, 0.4, 0.3, 0.1, 0.1, 0.1, 0.1]
        assert should_stop_early(loss_history, patience=3, min_delta=1e-4) is True

    def test_plateau_after_improvement(self):
        """Returns True when loss plateaus after initial improvement."""
        loss_history = [1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert should_stop_early(loss_history, patience=4, min_delta=0.01) is True

    def test_custom_patience_and_delta(self):
        """Custom patience and min_delta parameters work correctly."""
        # Loss improves by 0.003 in last 3 iterations, which is > min_delta of 0.001
        loss_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.599, 0.598, 0.597]
        assert should_stop_early(loss_history, patience=3, min_delta=0.001) is False
        # Loss improves by 0.001 in last 3 iterations (0.598 - 0.595 = 0.003), still > min_delta
        # Actually: best_before_window = 0.598, recent_best = 0.595, improvement = 0.003 > 0.001
        # So should NOT stop
        loss_history = [1.0, 0.9, 0.8, 0.7, 0.6, 0.599, 0.598, 0.597, 0.596, 0.595]
        assert should_stop_early(loss_history, patience=3, min_delta=0.001) is False

    def test_empty_loss_history(self):
        """Returns False for empty loss history."""
        assert should_stop_early([], patience=3, min_delta=0.001) is False


class TestAggregationModelWeightsWeightedAverageShape:
    """Tests for shape inconsistencies between nodes."""

    def test_shape_mismatch_within_layer(self):
        """Same number of layers but different shapes within a layer raises ValueError."""
        w1 = [np.array([[1.0, 2.0], [3.0, 4.0]])]  # shape (2, 2)
        w2 = [np.array([[1.0, 2.0, 3.0]])]  # shape (1, 3)
        results = [(w1, 10), (w2, 10)]
        with pytest.raises(ValueError):
            aggregation_model_weights_weighted_average(results)

    def test_dtype_mismatch_between_nodes(self):
        """Different dtypes between nodes - numpy promotes, but verify behaviour."""
        w1 = [np.array([1.0, 2.0], dtype=np.float32)]
        w2 = [np.array([3.0, 4.0], dtype=np.float64)]
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)
        # Result dtype should be float64 (promoted)
        assert aggregated[0].dtype == np.float64

    def test_negative_sample_count(self):
        """Negative sample count raises ValueError."""
        w = [np.array([1.0, 2.0])]
        results = [(w, -5)]
        # Negative sample counts should raise ValueError
        with pytest.raises(ValueError, match="Sample count must be non-negative"):
            aggregation_model_weights_weighted_average(results)

    def test_zero_total_samples(self):
        """Zero total samples across all nodes raises ValueError."""
        w1 = [np.array([1.0, 2.0, 3.0])]
        results = [(w1, 0)]
        # Zero total samples should raise ValueError
        with pytest.raises(ValueError, match="Total number of samples must be positive"):
            aggregation_model_weights_weighted_average(results)

    def test_all_zero_samples(self):
        """All nodes have zero samples - raises ValueError."""
        w1 = [np.array([1.0, 2.0])]
        w2 = [np.array([3.0, 4.0])]
        results = [(w1, 0), (w2, 0)]
        # Total samples = 0, should raise ValueError
        with pytest.raises(ValueError, match="Total number of samples must be positive"):
            aggregation_model_weights_weighted_average(results)


class TestAggregationDictWeights:
    """Tests for aggregation_model_weights_weighted_average with dict[str, np.ndarray] inputs."""

    def test_dict_aggregation_known_result(self):
        """Manually verify weighted average with dict inputs."""
        w1 = {"fc.weight": np.array([2.0, 4.0]), "fc.bias": np.array([1.0])}
        w2 = {"fc.weight": np.array([4.0, 8.0]), "fc.bias": np.array([3.0])}
        # 1 sample vs 3 samples
        results = [(w1, 1), (w2, 3)]
        aggregated = aggregation_model_weights_weighted_average(results)

        assert isinstance(aggregated, dict)
        np.testing.assert_array_almost_equal(aggregated["fc.weight"], np.array([3.5, 7.0]))
        np.testing.assert_array_almost_equal(aggregated["fc.bias"], np.array([2.5]))

    def test_dict_aggregation_preserves_keys(self):
        """Aggregated dict contains exactly the same keys as input nodes."""
        keys = ["encoder.weight", "encoder.bias", "decoder.weight"]
        w = {k: np.ones(3, dtype=np.float32) for k in keys}
        results = [(w, 10), (w, 10)]
        aggregated = aggregation_model_weights_weighted_average(results)

        assert isinstance(aggregated, dict)
        assert set(aggregated.keys()) == set(keys)

    def test_dict_aggregation_equal_nodes(self):
        """Equal weights from equal-sized dict nodes produce the same weights."""
        w = {"a": np.array([1.0, 2.0, 3.0])}
        results = [(w, 10), (w, 10)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated["a"], w["a"])

    def test_dict_aggregation_key_order_independent(self):
        """Aggregation with dict inputs is correct regardless of key insertion order."""
        w1 = {"x": np.array([1.0]), "y": np.array([2.0])}
        w2 = {"y": np.array([4.0]), "x": np.array([2.0])}  # reversed order
        results = [(w1, 1), (w2, 1)]
        aggregated = aggregation_model_weights_weighted_average(results)

        np.testing.assert_array_almost_equal(aggregated["x"], np.array([1.5]))
        np.testing.assert_array_almost_equal(aggregated["y"], np.array([3.0]))

    def test_dict_key_mismatch_raises(self):
        """Mismatched parameter names between nodes raise ValueError."""
        w1 = {"layer.weight": np.array([1.0, 2.0]), "layer.bias": np.array([0.1])}
        w2 = {"layer.weight": np.array([1.0, 2.0]), "WRONG.bias": np.array([0.1])}
        results = [(w1, 10), (w2, 10)]

        with pytest.raises(ValueError, match="mismatch"):
            aggregation_model_weights_weighted_average(results)

    def test_dict_extra_key_raises(self):
        """Node 2 having an extra key compared to node 1 raises ValueError."""
        w1 = {"a": np.array([1.0])}
        w2 = {"a": np.array([1.0]), "b": np.array([2.0])}
        results = [(w1, 10), (w2, 10)]

        with pytest.raises(ValueError):
            aggregation_model_weights_weighted_average(results)

    def test_dict_single_node(self):
        """A single dict-node's weights are returned unchanged."""
        w = {"p": np.array([5.0, 6.0, 7.0])}
        aggregated = aggregation_model_weights_weighted_average([(w, 42)])

        np.testing.assert_array_almost_equal(aggregated["p"], w["p"])


class TestReducelrOnPlateauStep:
    """Test cases for reducelr_on_plateau_step."""

    def test_initial_state_min_mode(self):
        """Initial state is set correctly for min mode."""
        # With metric=1.0 and mode="min", best is updated from inf to 1.0
        new_lr, state = reducelr_on_plateau_step(0.1, 1.0, {}, mode="min")
        assert new_lr == 0.1
        assert state["best"] == 1.0
        assert state["num_bad_epochs"] == 0

    def test_initial_state_max_mode(self):
        """Initial state is set correctly for max mode."""
        # With metric=1.0 and mode="max", best is updated from -inf to 1.0
        new_lr, state = reducelr_on_plateau_step(0.1, 1.0, {}, mode="max")
        assert new_lr == 0.1
        assert state["best"] == 1.0
        assert state["num_bad_epochs"] == 0

    def test_improving_metric_min_mode(self):
        """Metric improvement in min mode resets bad epochs counter."""
        state = {"best": 1.0, "num_bad_epochs": 5}
        new_lr, new_state = reducelr_on_plateau_step(0.1, 0.5, state, mode="min")
        assert new_lr == 0.1
        assert new_state["best"] == 0.5
        assert new_state["num_bad_epochs"] == 0

    def test_improving_metric_max_mode(self):
        """Metric improvement in max mode resets bad epochs counter."""
        state = {"best": 1.0, "num_bad_epochs": 5}
        new_lr, new_state = reducelr_on_plateau_step(0.1, 2.0, state, mode="max")
        assert new_lr == 0.1
        assert new_state["best"] == 2.0
        assert new_state["num_bad_epochs"] == 0

    def test_no_improvement_increments_bad_epochs(self):
        """No improvement increments the bad epochs counter."""
        state = {"best": 0.5, "num_bad_epochs": 2}
        new_lr, new_state = reducelr_on_plateau_step(0.1, 0.6, state, mode="min")
        assert new_lr == 0.1
        assert new_state["best"] == 0.5
        assert new_state["num_bad_epochs"] == 3

    def test_patience_exceeded_reduces_lr(self):
        """Exceeding patience reduces the learning rate."""
        state = {"best": 0.5, "num_bad_epochs": 10}
        new_lr, new_state = reducelr_on_plateau_step(
            0.1, 0.6, state, mode="min", patience=10, factor=0.1
        )
        assert new_lr == pytest.approx(0.01)
        assert new_state["num_bad_epochs"] == 0

    def test_min_lr_respected(self):
        """Learning rate is not reduced below min_lr."""
        state = {"best": 0.5, "num_bad_epochs": 10}
        new_lr, new_state = reducelr_on_plateau_step(
            0.001, 0.6, state, mode="min", patience=10, factor=0.1, min_lr=0.0005
        )
        assert new_lr == pytest.approx(0.0005)

    def test_eps_prevents_insignificant_updates(self):
        """Updates smaller than eps are ignored."""
        state = {"best": 0.5, "num_bad_epochs": 10}
        # 0.1 * 0.1 = 0.01, difference is 0.09 > eps (1e-8)
        new_lr, new_state = reducelr_on_plateau_step(
            0.1, 0.6, state, mode="min", patience=10, factor=0.1, eps=0.1
        )
        # Should not update because 0.1 - 0.01 = 0.09 < eps=0.1
        assert new_lr == 0.1
        assert new_state["num_bad_epochs"] == 0

    def test_relative_threshold_min_mode(self):
        """Relative threshold is computed correctly in min mode."""
        state = {"best": 1.0, "num_bad_epochs": 0}
        # threshold = 0.1, rel mode: metric < best * (1 - 0.1) = 1.0 * 0.9 = 0.9
        new_lr, new_state = reducelr_on_plateau_step(
            0.1, 0.89, state, mode="min", threshold=0.1, threshold_mode="rel"
        )
        # 0.89 < 0.9, so should be better
        assert new_state["num_bad_epochs"] == 0
        assert new_state["best"] == 0.89

    def test_absolute_threshold_min_mode(self):
        """Absolute threshold is computed correctly in min mode."""
        state = {"best": 1.0, "num_bad_epochs": 0}
        # threshold = 0.1, abs mode: metric < best - 0.1 = 0.9
        new_lr, new_state = reducelr_on_plateau_step(
            0.1, 0.89, state, mode="min", threshold=0.1, threshold_mode="abs"
        )
        # 0.89 < 0.9, so should be better
        assert new_state["num_bad_epochs"] == 0
        assert new_state["best"] == 0.89

    def test_invalid_mode_raises(self):
        """Invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode 'invalid' is unknown"):
            reducelr_on_plateau_step(0.1, 1.0, {}, mode="invalid")

    def test_invalid_threshold_mode_raises(self):
        """Invalid threshold_mode raises ValueError."""
        with pytest.raises(ValueError, match="threshold_mode 'invalid' is unknown"):
            reducelr_on_plateau_step(0.1, 1.0, {}, threshold_mode="invalid")

    def test_invalid_factor_raises(self):
        """Factor >= 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Factor should be < 1.0"):
            reducelr_on_plateau_step(0.1, 1.0, {}, factor=1.0)

    def test_multiple_reductions(self):
        """Multiple reductions work correctly in sequence."""
        state = {}
        lr = 1.0

        # First call - initial state, metric improves from inf to 1.0
        lr, state = reducelr_on_plateau_step(lr, 1.0, state, mode="min", patience=1)
        assert lr == 1.0

        # Second call - no improvement (1.0 not < 1.0), num_bad_epochs becomes 1
        # but patience=1, so need one more call to exceed patience
        lr, state = reducelr_on_plateau_step(lr, 1.0, state, mode="min", patience=1, factor=0.5)
        assert lr == 1.0  # Not reduced yet, num_bad_epochs=1 not > patience=1

        # Third call - no improvement, num_bad_epochs becomes 2, which > patience=1
        # LR reduces: 1.0 -> 0.5; num_bad_epochs resets to 0
        lr, state = reducelr_on_plateau_step(lr, 1.0, state, mode="min", patience=1, factor=0.5)
        assert lr == pytest.approx(0.5)

        # Fourth call - no improvement, num_bad_epochs becomes 1 (not > patience=1, no reduction yet)
        lr, state = reducelr_on_plateau_step(lr, 1.0, state, mode="min", patience=1, factor=0.5)
        assert lr == pytest.approx(0.5)

        # Fifth call - no improvement, num_bad_epochs becomes 2, which > patience=1
        # LR reduces: 0.5 -> 0.25; num_bad_epochs resets to 0
        lr, state = reducelr_on_plateau_step(lr, 1.0, state, mode="min", patience=1, factor=0.5)
        assert lr == pytest.approx(0.25)

    def test_plateau_behaviour(self):
        """Plateau (no change) is treated as no improvement."""
        state = {"best": 0.5, "num_bad_epochs": 5}
        new_lr, new_state = reducelr_on_plateau_step(
            0.1, 0.5, state, mode="min", patience=5, factor=0.1
        )
        # Same metric as best -> no improvement -> bad epochs increments
        # Then patience exceeded -> LR reduced
        assert new_lr == pytest.approx(0.01)


class TestLrDetermination:
    """Test cases for lr_determination."""

    def test_reducelr_on_plateau_scheduler(self):
        """lr_determination works with reducelr_on_plateau scheduler."""
        new_lr, state = lr_determination(0.1, 1.0, scheduler="reducelr_on_plateau", state=None)
        assert new_lr == 0.1
        assert "best" in state
        assert "num_bad_epochs" in state

    def test_unsupported_scheduler_raises(self):
        """Unsupported scheduler raises ValueError."""
        with pytest.raises(ValueError, match="Scheduler 'unknown' is not supported"):
            lr_determination(0.1, 1.0, scheduler="unknown")

    def test_passes_kwargs_to_scheduler(self):
        """Additional kwargs are passed to the scheduler."""
        new_lr, state = lr_determination(
            0.1,
            1.0,
            scheduler="reducelr_on_plateau",
            state=None,
            mode="max",
            factor=0.5,
            patience=5,
        )
        assert new_lr == 0.1
        assert state["best"] == 1.0  # max mode: -inf updated to 1.0 because 1.0 > -inf

    def test_state_persists_between_calls(self):
        """State persists correctly between calls."""
        state = None

        # First call - metric improves from inf to 1.0
        lr, state = lr_determination(
            1.0, 1.0, scheduler="reducelr_on_plateau", state=state, patience=1
        )
        assert lr == 1.0

        # Second call - no improvement, num_bad_epochs=1, not > patience=1 yet
        lr, state = lr_determination(
            lr, 1.0, scheduler="reducelr_on_plateau", state=state, patience=1, factor=0.5
        )
        assert lr == 1.0  # Not reduced yet

        # Third call - no improvement, num_bad_epochs=2 > patience=1, should reduce LR
        lr, state = lr_determination(
            lr, 1.0, scheduler="reducelr_on_plateau", state=state, patience=1, factor=0.5
        )
        assert lr == pytest.approx(0.5)

    def test_default_scheduler_is_reducelr_on_plateau(self):
        """Default scheduler is reducelr_on_plateau."""
        new_lr, state = lr_determination(0.1, 1.0)
        assert new_lr == 0.1
        assert "best" in state
