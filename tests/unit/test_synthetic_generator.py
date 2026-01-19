"""
Unit tests for synthetic data generator functionality.
"""

import pytest
import pandas as pd
import numpy as np
from fed_synthetic_data.synthetic_generator import (
    TabularSyntheticGenerator,
    train_federated_generator,
)


@pytest.mark.unit
class TestTabularSyntheticGenerator:
    """Test cases for TabularSyntheticGenerator class."""

    def test_initialization(self):
        """Test that generator initializes with correct parameters."""
        generator = TabularSyntheticGenerator(
            model_type="tabular",
            privacy_mechanism="dp",
            epsilon=1.0
        )
        
        assert generator.model_type == "tabular"
        assert generator.privacy_mechanism == "dp"
        assert generator.epsilon == 1.0
        assert not generator.is_fitted

    def test_initialization_defaults(self):
        """Test that generator initializes with default parameters."""
        generator = TabularSyntheticGenerator()
        
        assert generator.model_type == "tabular"
        assert generator.privacy_mechanism is None
        assert generator.epsilon == 1.0
        assert not generator.is_fitted

    def test_fit_sets_fitted_flag(self, sample_tabular_data):
        """Test that fit method sets the fitted flag."""
        generator = TabularSyntheticGenerator()
        
        assert not generator.is_fitted
        generator.fit(sample_tabular_data)
        assert generator.is_fitted

    def test_fit_returns_self(self, sample_tabular_data):
        """Test that fit method returns self for chaining."""
        generator = TabularSyntheticGenerator()
        result = generator.fit(sample_tabular_data)
        
        assert result is generator

    def test_sample_requires_fitted_model(self):
        """Test that sampling requires a fitted model."""
        generator = TabularSyntheticGenerator()
        
        with pytest.raises(ValueError, match="must be fitted"):
            generator.sample(n_samples=100)

    def test_sample_after_fit(self, sample_tabular_data):
        """Test that sampling works after fitting."""
        generator = TabularSyntheticGenerator()
        generator.fit(sample_tabular_data)
        
        # Should not raise an error
        result = generator.sample(n_samples=100)
        assert isinstance(result, pd.DataFrame)

    def test_evaluate_returns_dict(self, sample_tabular_data):
        """Test that evaluate returns a dictionary."""
        generator = TabularSyntheticGenerator()
        result = generator.evaluate(sample_tabular_data, sample_tabular_data)
        
        assert isinstance(result, dict)


@pytest.mark.unit
class TestTrainFederatedGenerator:
    """Test cases for train_federated_generator function."""

    def test_basic_training(self, sample_tabular_data):
        """Test basic federated generator training."""
        generator = train_federated_generator(sample_tabular_data)
        
        assert isinstance(generator, TabularSyntheticGenerator)
        assert generator.is_fitted

    def test_training_with_config(self, sample_tabular_data):
        """Test training with custom configuration."""
        model_config = {"model_type": "tabular"}
        privacy_config = {"privacy_mechanism": "dp", "epsilon": 0.5}
        
        generator = train_federated_generator(
            sample_tabular_data,
            model_config=model_config,
            privacy_config=privacy_config
        )
        
        assert generator.model_type == "tabular"
        assert generator.privacy_mechanism == "dp"
        assert generator.epsilon == 0.5

    def test_training_with_empty_config(self, sample_tabular_data):
        """Test that training works with None configs."""
        generator = train_federated_generator(
            sample_tabular_data,
            model_config=None,
            privacy_config=None
        )
        
        assert isinstance(generator, TabularSyntheticGenerator)
        assert generator.is_fitted


@pytest.mark.unit
class TestGeneratorEdgeCases:
    """Test edge cases for synthetic data generator."""

    def test_fit_with_minimal_data(self):
        """Test fitting with minimal data."""
        minimal_data = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["A", "B", "C"]
        })
        
        generator = TabularSyntheticGenerator()
        generator.fit(minimal_data)
        
        assert generator.is_fitted

    def test_sample_with_seed_data(self, sample_tabular_data):
        """Test sampling with seed data."""
        generator = TabularSyntheticGenerator()
        generator.fit(sample_tabular_data)
        
        seed_data = pd.DataFrame({
            "age": [25, 30],
            "gender": ["Male", "Female"]
        })
        
        result = generator.sample(seed_data=seed_data)
        assert isinstance(result, pd.DataFrame)
