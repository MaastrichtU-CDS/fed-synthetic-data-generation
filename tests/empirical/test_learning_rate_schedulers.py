"""
Empirical tests for learning rate scheduler functions.

These tests verify that the learning rate determination functions
produce mathematically correct results and behave as expected in
realistic federated training scenarios.
"""

import math

import pytest

from fed_synthetic_data.federated_training import (
    lr_determination,
    reducelr_on_plateau_step,
)


class TestReduceLrOnPlateauMathematicalCorrectness:
    """Verify ReduceLROnPlateau mathematical correctness against known PyTorch behaviour."""

    def test_exact_pytorch_behaviour_min_mode(self):
        """Test that our implementation matches PyTorch's ReduceLROnPlateau for min mode."""
        # Simulate PyTorch's ReduceLROnPlateau with mode='min'
        # Initial: lr=0.1, best=inf, num_bad_epochs=0
        # Step 1: metric=1.0 -> no improvement (1.0 > inf is False, but inf comparison is special)
        # Actually: metric=1.0, best=inf -> 1.0 < inf is True -> improvement
        state = {}
        lr = 0.1

        # First step: metric improves from inf to 1.0
        lr, state = reducelr_on_plateau_step(lr, 1.0, state, mode="min")
        assert lr == 0.1
        assert state["best"] == 1.0
        assert state["num_bad_epochs"] == 0

        # Second step: metric=0.9 (improvement)
        lr, state = reducelr_on_plateau_step(lr, 0.9, state, mode="min")
        assert lr == 0.1
        assert state["best"] == 0.9
        assert state["num_bad_epochs"] == 0

        # Third step: metric=0.91 (no improvement, within threshold)
        # With default threshold=1e-4, threshold_mode='rel':
        # is_better = 0.91 < 0.9 * (1 - 1e-4) = 0.9 * 0.9999 = 0.89991
        # 0.91 < 0.89991 is False -> no improvement
        lr, state = reducelr_on_plateau_step(lr, 0.91, state, mode="min")
        assert lr == 0.1
        assert state["best"] == 0.9
        assert state["num_bad_epochs"] == 1

    def test_threshold_relative_mode_calculation(self):
        """Verify relative threshold calculation is mathematically correct."""
        # With threshold=0.1, threshold_mode='rel', mode='min':
        # is_better = metric < best * (1 - threshold)
        state = {"best": 100.0, "num_bad_epochs": 0}
        lr = 1.0

        # metric = 90.0, threshold = 0.1
        # best * (1 - threshold) = 100 * 0.9 = 90.0
        # metric < 90.0 is False, so 90.0 is NOT better than 100.0
        lr, state = reducelr_on_plateau_step(
            lr, 90.0, state, mode="min", threshold=0.1, threshold_mode="rel"
        )
        assert state["num_bad_epochs"] == 1  # Not better

        # metric = 89.9
        # 89.9 < 90.0 is True, so it IS better
        state = {"best": 100.0, "num_bad_epochs": 0}
        lr, state = reducelr_on_plateau_step(
            lr, 89.9, state, mode="min", threshold=0.1, threshold_mode="rel"
        )
        assert state["num_bad_epochs"] == 0  # Better
        assert state["best"] == 89.9

    def test_threshold_absolute_mode_calculation(self):
        """Verify absolute threshold calculation is mathematically correct."""
        # With threshold=0.5, threshold_mode='abs', mode='min':
        # is_better = metric < best - threshold
        state = {"best": 10.0, "num_bad_epochs": 0}
        lr = 1.0

        # metric = 9.5
        # best - threshold = 10.0 - 0.5 = 9.5
        # metric < 9.5 is False, so 9.5 is NOT better
        lr, state = reducelr_on_plateau_step(
            lr, 9.5, state, mode="min", threshold=0.5, threshold_mode="abs"
        )
        assert state["num_bad_epochs"] == 1

        # metric = 9.4
        # 9.4 < 9.5 is True, so it IS better
        state = {"best": 10.0, "num_bad_epochs": 0}
        lr, state = reducelr_on_plateau_step(
            lr, 9.4, state, mode="min", threshold=0.5, threshold_mode="abs"
        )
        assert state["num_bad_epochs"] == 0
        assert state["best"] == 9.4

    def test_max_mode_mathematics(self):
        """Verify max mode mathematics are correct."""
        # mode='max': we want to MAXIMISE the metric (e.g. accuracy)
        # is_better = metric > best * (1 + threshold) for rel mode
        state = {"best": 0.5, "num_bad_epochs": 0}
        lr = 1.0

        # metric = 0.51, threshold = 0.1, rel mode
        # best * (1 + threshold) = 0.5 * 1.1 = 0.55
        # metric > 0.55 is False, so NOT better
        lr, state = reducelr_on_plateau_step(
            lr, 0.51, state, mode="max", threshold=0.1, threshold_mode="rel"
        )
        assert state["num_bad_epochs"] == 1

        # metric = 0.56
        # 0.56 > 0.55 is True, so it IS better
        state = {"best": 0.5, "num_bad_epochs": 0}
        lr, state = reducelr_on_plateau_step(
            lr, 0.56, state, mode="max", threshold=0.1, threshold_mode="rel"
        )
        assert state["num_bad_epochs"] == 0
        assert state["best"] == 0.56

    def test_learning_rate_reduction_formula(self):
        """Verify the learning rate reduction formula is correct."""
        # new_lr = max(current_lr * factor, min_lr)
        state = {"best": 1.0, "num_bad_epochs": 10}
        current_lr = 0.1

        # With factor=0.5, min_lr=0.0
        lr, _ = reducelr_on_plateau_step(
            current_lr, 1.0, state, mode="min", patience=10, factor=0.5, min_lr=0.0
        )
        expected = 0.1 * 0.5  # = 0.05
        assert lr == pytest.approx(expected)

        # With factor=0.5, min_lr=0.01
        state = {"best": 1.0, "num_bad_epochs": 10}
        lr, _ = reducelr_on_plateau_step(
            current_lr, 1.0, state, mode="min", patience=10, factor=0.5, min_lr=0.01
        )
        expected = max(0.1 * 0.5, 0.01)  # max(0.05, 0.01) = 0.05
        assert lr == pytest.approx(expected)

        # With factor=0.1, min_lr=0.01
        state = {"best": 1.0, "num_bad_epochs": 10}
        lr, _ = reducelr_on_plateau_step(
            current_lr, 1.0, state, mode="min", patience=10, factor=0.1, min_lr=0.01
        )
        expected = max(0.1 * 0.1, 0.01)  # max(0.01, 0.01) = 0.01
        assert lr == pytest.approx(expected)

    def test_learning_rate_reduction_with_epsilon(self):
        """Verify epsilon prevents insignificant updates."""
        state = {"best": 1.0, "num_bad_epochs": 10}
        current_lr = 0.1

        # Without epsilon: 0.1 * 0.1 = 0.01, difference = 0.09
        # With eps=0.1: 0.09 < 0.1, so update is ignored
        lr, state = reducelr_on_plateau_step(
            current_lr, 1.0, state, mode="min", patience=10, factor=0.1, eps=0.1
        )
        assert lr == pytest.approx(0.1)  # Not changed
        assert state["num_bad_epochs"] == 0  # Still reset because we attempted

        # With eps=0.05: 0.09 > 0.05, so update is applied
        state = {"best": 1.0, "num_bad_epochs": 10}
        lr, state = reducelr_on_plateau_step(
            current_lr, 1.0, state, mode="min", patience=10, factor=0.1, eps=0.05
        )
        assert lr == pytest.approx(0.01)  # Changed


class TestFederatedTrainingScenario:
    """Test realistic federated training scenarios."""

    def test_learning_rate_stays_constant_with_improving_loss(self):
        """LR stays constant when loss improves every round."""
        state = None
        lr = 0.1

        # Simulate 10 rounds with improving loss
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for loss in losses:
            lr, state = lr_determination(lr, loss, scheduler="reducelr_on_plateau", state=state)

        # LR should remain at 0.1 since loss is always improving
        assert lr == pytest.approx(0.1)

    def test_learning_rate_reduces_on_plateau(self):
        """LR reduces when loss plateaus."""
        state = None
        lr = 0.1

        # Simulate: improving for 5 rounds, then plateau for 6 rounds
        # With patience=5, LR should reduce after the 6th plateau round
        # Need: 5 improving + 6 plateau = 11 total
        # First 5 improve best, then 6 plateaus
        # After 5 plateaus (total 10), num_bad_epochs=5, which is NOT > 5
        # After 6 plateaus (total 11), num_bad_epochs=6 > 5, so LR reduces
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]  # improving
        losses += [0.6] * 6  # plateau - need all 6 to exceed patience

        for loss in losses:
            lr, state = lr_determination(
                lr, loss, scheduler="reducelr_on_plateau", state=state, patience=5, factor=0.5
            )

        # After the plateau, LR should have been reduced
        assert lr == pytest.approx(0.05)

    def test_multiple_learning_rate_reductions(self):
        """Multiple LR reductions in a long training run."""
        state = None
        lr = 1.0
        factor = 0.1

        # First phase: improving
        for loss in [10.0, 9.0, 8.0, 7.0, 6.0]:
            lr, state = lr_determination(
                lr, loss, scheduler="reducelr_on_plateau", state=state, patience=2, factor=factor
            )

        # Second phase: plateau (need patience+1=3 iterations to trigger reduction)
        # After 5 improving: best=6.0, num_bad_epochs=0
        # plateau call 1: num_bad_epochs=1
        # plateau call 2: num_bad_epochs=2
        # plateau call 3: num_bad_epochs=3 > patience=2 -> LR reduces
        for loss in [6.0, 6.0, 6.0, 6.0]:
            lr, state = lr_determination(
                lr, loss, scheduler="reducelr_on_plateau", state=state, patience=2, factor=factor
            )

        # LR should have been reduced once: 1.0 -> 0.1
        assert lr == pytest.approx(0.1)

        # Third phase: more plateau (3 more iterations)
        for loss in [6.0, 6.0, 6.0]:
            lr, state = lr_determination(
                lr, loss, scheduler="reducelr_on_plateau", state=state, patience=2, factor=factor
            )

        # LR should have been reduced again: 0.1 -> 0.01
        assert lr == pytest.approx(0.01)

    def test_learning_rate_hits_minimum(self):
        """LR stops reducing when it hits min_lr."""
        state = None
        lr = 1.0
        min_lr = 0.001

        # Need patience+1 calls with no improvement to trigger reduction
        # First reduction: 1.0 -> 0.1 (need 11 calls: 1 to set best, 10+1 to exceed patience)
        # But first call sets best=1.0, then 11 more calls with same metric
        # After 11 calls total: first sets best, then 10 accumulate num_bad_epochs
        # On the 11th call (10th with no improvement), num_bad_epochs=10, which is NOT > 10
        # Need 12 calls total: 1 to set best, 11 with no improvement
        for i, loss in enumerate([1.0] * 12):
            lr, state = lr_determination(
                lr,
                loss,
                scheduler="reducelr_on_plateau",
                state=state,
                patience=10,
                factor=0.1,
                min_lr=min_lr,
            )
            if i == 0:
                # First call sets best=1.0
                assert state["best"] == 1.0
                assert state["num_bad_epochs"] == 0

        assert lr == pytest.approx(0.1)

        # Second reduction: 0.1 -> 0.01
        for loss in [1.0] * 11:
            lr, state = lr_determination(
                lr,
                loss,
                scheduler="reducelr_on_plateau",
                state=state,
                patience=10,
                factor=0.1,
                min_lr=min_lr,
            )

        assert lr == pytest.approx(0.01)

        # Third reduction: 0.01 -> 0.001 (hits min_lr)
        for loss in [1.0] * 11:
            lr, state = lr_determination(
                lr,
                loss,
                scheduler="reducelr_on_plateau",
                state=state,
                patience=10,
                factor=0.1,
                min_lr=min_lr,
            )

        assert lr == pytest.approx(0.001)

        # Fourth attempt: should stay at min_lr
        for loss in [1.0] * 11:
            lr, state = lr_determination(
                lr,
                loss,
                scheduler="reducelr_on_plateau",
                state=state,
                patience=10,
                factor=0.1,
                min_lr=min_lr,
            )

        assert lr == pytest.approx(0.001)


class TestNumericalStability:
    """Test numerical stability with extreme values."""

    def test_very_small_learning_rates(self):
        """Very small learning rates are handled correctly."""
        lr = 1e-10
        # Need num_bad_epochs > patience (10) to trigger reduction.
        # eps=0.0 is required: the default eps (1e-8) would block the update because
        # the difference 1e-10 - 1e-11 = 9e-11 is smaller than eps=1e-8.
        state = {"best": 1.0, "num_bad_epochs": 11}

        new_lr, _ = reducelr_on_plateau_step(
            lr, 1.0, state, mode="min", patience=10, factor=0.1, min_lr=0.0, eps=0.0
        )

        expected = 1e-11
        assert new_lr == pytest.approx(expected)

    def test_very_large_metrics(self):
        """Very large metric values don't cause overflow."""
        lr = 0.1
        metric = 1e10
        state = {}

        new_lr, state = reducelr_on_plateau_step(lr, metric, state, mode="min")
        assert new_lr == 0.1
        assert state["best"] == 1e10

    def test_zero_learning_rate(self):
        """Zero learning rate is handled correctly."""
        lr = 0.0
        state = {"best": 1.0, "num_bad_epochs": 10}

        new_lr, _ = reducelr_on_plateau_step(lr, 1.0, state, mode="min", patience=10, factor=0.1)
        assert new_lr == 0.0

    def test_negative_metric_values(self):
        """Negative metric values are handled correctly in min mode."""
        # In min mode, we want to minimise the metric
        # Negative metrics are valid (e.g., negative log-likelihood)
        lr = 0.1
        state = {}

        # First call with negative metric
        new_lr, state = reducelr_on_plateau_step(lr, -1.0, state, mode="min")
        assert new_lr == 0.1
        assert state["best"] == -1.0

        # Second call with more negative (better) metric
        new_lr, state = reducelr_on_plateau_step(lr, -2.0, state, mode="min")
        assert new_lr == 0.1
        assert state["best"] == -2.0
        assert state["num_bad_epochs"] == 0

        # Third call with less negative (worse) metric
        new_lr, state = reducelr_on_plateau_step(lr, -1.5, state, mode="min")
        assert new_lr == 0.1
        assert state["best"] == -2.0
        assert state["num_bad_epochs"] == 1
