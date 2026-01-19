"""
Unit tests for privacy measures functionality.
"""

import pytest
import numpy as np
import pandas as pd
from fed_synthetic_data.privacy_measures import (
    apply_differential_privacy,
    compute_privacy_budget,
    add_laplace_noise,
    add_gaussian_noise,
    clip_gradients,
    compute_privacy_loss,
)


@pytest.mark.unit
class TestApplyDifferentialPrivacy:
    """Test cases for apply_differential_privacy function."""

    def test_with_dataframe(self, sample_numerical_data):
        """Test applying DP to a DataFrame."""
        result = apply_differential_privacy(
            sample_numerical_data,
            epsilon=1.0,
            mechanism="laplace"
        )
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_numerical_data.shape

    def test_with_numpy_array(self):
        """Test applying DP to a numpy array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_differential_privacy(data, epsilon=1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == data.shape

    def test_raises_on_invalid_epsilon(self, sample_numerical_data):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            apply_differential_privacy(sample_numerical_data, epsilon=0)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            apply_differential_privacy(sample_numerical_data, epsilon=-1.0)

    def test_raises_on_invalid_delta(self, sample_numerical_data):
        """Test that invalid delta raises error."""
        with pytest.raises(ValueError, match="Delta must be in"):
            apply_differential_privacy(sample_numerical_data, delta=-0.1)
        
        with pytest.raises(ValueError, match="Delta must be in"):
            apply_differential_privacy(sample_numerical_data, delta=1.0)


@pytest.mark.unit
class TestComputePrivacyBudget:
    """Test cases for compute_privacy_budget function."""

    def test_basic_composition(self):
        """Test basic privacy composition."""
        total = compute_privacy_budget(10, 0.1, composition_method="basic")
        assert total == 1.0

    def test_advanced_composition(self):
        """Test advanced composition."""
        total = compute_privacy_budget(10, 0.1, composition_method="advanced")
        assert total < 1.0  # Should be less than basic

    def test_rdp_composition(self):
        """Test RDP composition."""
        total = compute_privacy_budget(10, 0.1, composition_method="rdp")
        assert total < 1.0  # Should be less than basic

    def test_raises_on_invalid_queries(self):
        """Test that invalid number of queries raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_privacy_budget(0, 1.0)
        
        with pytest.raises(ValueError, match="must be positive"):
            compute_privacy_budget(-5, 1.0)

    def test_raises_on_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ValueError, match="Epsilon per query must be positive"):
            compute_privacy_budget(10, 0)

    def test_raises_on_unknown_method(self):
        """Test that unknown composition method raises error."""
        with pytest.raises(ValueError, match="Unknown composition method"):
            compute_privacy_budget(10, 0.1, composition_method="unknown")


@pytest.mark.unit
class TestAddLaplaceNoise:
    """Test cases for add_laplace_noise function."""

    def test_adds_noise_to_float(self):
        """Test adding Laplace noise to a float."""
        value = 10.0
        result = add_laplace_noise(value, sensitivity=1.0, epsilon=1.0)
        
        assert isinstance(result, float)
        # Should be different due to noise (with very high probability)
        assert result != value

    def test_adds_noise_to_array(self):
        """Test adding Laplace noise to an array."""
        value = np.array([1.0, 2.0, 3.0])
        result = add_laplace_noise(value, sensitivity=1.0, epsilon=1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == value.shape

    def test_raises_on_invalid_sensitivity(self):
        """Test that invalid sensitivity raises error."""
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            add_laplace_noise(10.0, sensitivity=0, epsilon=1.0)

    def test_raises_on_invalid_epsilon(self):
        """Test that invalid epsilon raises error."""
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            add_laplace_noise(10.0, sensitivity=1.0, epsilon=0)


@pytest.mark.unit
class TestAddGaussianNoise:
    """Test cases for add_gaussian_noise function."""

    def test_adds_noise_to_float(self):
        """Test adding Gaussian noise to a float."""
        value = 10.0
        result = add_gaussian_noise(value, sensitivity=1.0, epsilon=1.0)
        
        assert isinstance(result, float)

    def test_adds_noise_to_array(self):
        """Test adding Gaussian noise to an array."""
        value = np.array([1.0, 2.0, 3.0])
        result = add_gaussian_noise(value, sensitivity=1.0, epsilon=1.0)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == value.shape

    def test_raises_on_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            add_gaussian_noise(10.0, sensitivity=0, epsilon=1.0)
        
        with pytest.raises(ValueError):
            add_gaussian_noise(10.0, sensitivity=1.0, epsilon=0)
        
        with pytest.raises(ValueError):
            add_gaussian_noise(10.0, sensitivity=1.0, epsilon=1.0, delta=0)
        
        with pytest.raises(ValueError):
            add_gaussian_noise(10.0, sensitivity=1.0, epsilon=1.0, delta=1.0)


@pytest.mark.unit
class TestClipGradients:
    """Test cases for clip_gradients function."""

    def test_clips_large_gradients(self):
        """Test clipping of large gradients."""
        gradients = np.array([10.0, 10.0, 10.0])
        max_norm = 5.0
        
        result = clip_gradients(gradients, max_norm)
        
        result_norm = np.linalg.norm(result)
        assert result_norm <= max_norm

    def test_does_not_clip_small_gradients(self):
        """Test that small gradients are not clipped."""
        gradients = np.array([1.0, 1.0, 1.0])
        max_norm = 10.0
        
        result = clip_gradients(gradients, max_norm)
        
        np.testing.assert_allclose(result, gradients)

    def test_raises_on_invalid_max_norm(self):
        """Test that invalid max_norm raises error."""
        gradients = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="Max norm must be positive"):
            clip_gradients(gradients, max_norm=0)
        
        with pytest.raises(ValueError, match="Max norm must be positive"):
            clip_gradients(gradients, max_norm=-1.0)


@pytest.mark.unit
class TestComputePrivacyLoss:
    """Test cases for compute_privacy_loss function."""

    def test_computes_privacy_loss(self):
        """Test computing privacy loss metrics."""
        result = compute_privacy_loss(epsilon=0.1, delta=1e-5, num_rounds=10)
        
        assert isinstance(result, dict)
        assert "epsilon_per_round" in result
        assert "total_epsilon" in result
        assert "delta" in result
        assert "num_rounds" in result
        assert "privacy_guarantee" in result

    def test_correct_values(self):
        """Test that computed values are correct."""
        epsilon = 0.1
        delta = 1e-5
        num_rounds = 10
        
        result = compute_privacy_loss(epsilon, delta, num_rounds)
        
        assert result["epsilon_per_round"] == epsilon
        assert result["delta"] == delta
        assert result["num_rounds"] == num_rounds
        assert result["total_epsilon"] == num_rounds * epsilon
