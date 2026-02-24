"""
Unit tests for federated_training module.

This module contains unit tests for the federated training functions,
including train_single_iteration and extract_model_weights.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from mostlyai.engine import TabularARGN

from fed_synthetic_data.federated_training import (
    train_single_iteration,
    extract_model_weights,
    EPOCHS_PER_ITERATION,
)


class TestTrainSingleIteration:
    """Test cases for the train_single_iteration function."""

    def test_train_single_iteration_with_new_model(self):
        """Test training with a new model (None provided)."""
        # Create sample data
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["A", "B", "C", "D", "E"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        # Mock the TabularARGN class
        with patch("fed_synthetic_data.federated_training.TabularARGN") as mock_tabular_argn:
            mock_model_instance = Mock()
            mock_tabular_argn.return_value = mock_model_instance
            mock_tabular_argn.max_epochs = EPOCHS_PER_ITERATION

            # Call the function
            result = train_single_iteration(data)

            # Verify the model was created with correct max_epochs
            mock_tabular_argn.assert_called_once_with(max_epochs=EPOCHS_PER_ITERATION)

            # Verify fit was called with the data
            mock_model_instance.fit.assert_called_once_with(data)

            # Verify the result is the mock model instance
            assert result == mock_model_instance

    def test_train_single_iteration_with_existing_model(self):
        """Test training with an existing model."""
        # Create sample data
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["A", "B", "C", "D", "E"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        # Create a mock for an existing model
        mock_model = Mock(spec=TabularARGN)
        mock_model.max_epochs = EPOCHS_PER_ITERATION

        # Call the function
        result = train_single_iteration(data, mock_model)

        # Verify fit was called with the data
        mock_model.fit.assert_called_once_with(data)

        # Verify the result is the same model instance
        assert result == mock_model

    def test_train_single_iteration_with_different_max_epochs(self):
        """Test training with a model having different max_epochs."""
        # Create sample data
        data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["A", "B", "C", "D", "E"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        # Create a mock existing model with different max_epochs
        mock_model = Mock(spec=TabularARGN)
        mock_model.max_epochs = 10  # Different from EPOCHS_PER_ITERATION

        # Call the function - should still work but not modify max_epochs
        result = train_single_iteration(data, mock_model)

        # Verify fit was called with the data
        mock_model.fit.assert_called_once_with(data)

        # Verify the result is the same model instance
        assert result == mock_model

    def test_train_single_iteration_empty_data(self):
        """Test training with empty data."""
        # Create empty data
        data = pd.DataFrame()

        # Mock the TabularARGN class
        with patch("fed_synthetic_data.federated_training.TabularARGN") as mock_tabular_argn:
            mock_model_instance = Mock()
            mock_tabular_argn.return_value = mock_model_instance

            # Call the function - should still work
            result = train_single_iteration(data)

            # Verify the model was created
            mock_tabular_argn.assert_called_once_with(max_epochs=EPOCHS_PER_ITERATION)

            # Verify fit was called with the empty data
            mock_model_instance.fit.assert_called_once_with(data)


class TestConstants:
    """Test cases for module constants."""

    def test_epochs_per_iteration_value(self):
        """Test that EPOCHS_PER_ITERATION has the expected value."""
        assert EPOCHS_PER_ITERATION >= 1
        assert isinstance(EPOCHS_PER_ITERATION, int)
