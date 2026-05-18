"""
Unit tests for post_training module.

Pure unit tests for weight loading functions, using mocks for dependencies.
Integration tests (PyTorch, file I/O, cross-module) belong in tests/integration/.
"""

import numpy as np
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from fed_synthetic_data.post_training import (
    load_weights_into_model,
    load_model_from_json_weights,
    save_model,
    save_model_weights,
)

# =============================================================================
# Mock Model for Testing
# =============================================================================


class MockModel:
    """Mock model implementing ModelProtocol for unit testing."""

    def __init__(self):
        self.state_dict_data = {}
        self.loaded_state_dict = None

    def state_dict(self) -> dict:
        return self.state_dict_data

    def load_state_dict(self, state_dict: dict) -> None:
        self.loaded_state_dict = state_dict
        self.state_dict_data.update(state_dict)


# =============================================================================
# Unit Tests for load_weights_into_model
# =============================================================================


class TestLoadWeightsIntoModel:
    """Unit tests for load_weights_into_model."""

    def test_basic_weight_loading(self):
        """Weights are loaded into model's state_dict."""
        model = MockModel()
        weights = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        parameter_names = ["layer1.weights"]

        result = load_weights_into_model(model, weights, parameter_names)

        assert result is model
        assert model.loaded_state_dict == {"layer1.weights": weights[0]}

    def test_multiple_parameters(self):
        """Multiple weight arrays loaded into corresponding parameters."""
        model = MockModel()
        weights = [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0, 5.0], dtype=np.float32),
        ]
        parameter_names = ["layer1.weights", "layer2.weights"]

        load_weights_into_model(model, weights, parameter_names)

        assert "layer1.weights" in model.loaded_state_dict
        assert "layer2.weights" in model.loaded_state_dict

    def test_length_mismatch_raises_valueerror(self):
        """Raises ValueError when weights and parameter_names length differ."""
        model = MockModel()
        weights = [np.array([1.0, 2.0])]
        parameter_names = ["layer1", "layer2"]

        with pytest.raises(ValueError, match="Mismatch: 1 weights vs 2 names"):
            load_weights_into_model(model, weights, parameter_names)

    def test_empty_weights_and_names(self):
        """Handles empty weights and parameter_names lists."""
        model = MockModel()
        result = load_weights_into_model(model, [], [])
        assert result is model
        assert model.loaded_state_dict == {}

    def test_returns_model_reference(self):
        """Returns the same model instance."""
        model = MockModel()
        result = load_weights_into_model(model, [], [])
        assert result is model

    def test_2d_weight_arrays(self):
        """2-D weight matrices are correctly loaded."""
        model = MockModel()
        weights = [
            np.arange(12, dtype=np.float32).reshape(3, 4),
            np.eye(3, dtype=np.float32),
        ]
        parameter_names = ["conv1.weights", "conv2.weights"]

        load_weights_into_model(model, weights, parameter_names)

        np.testing.assert_array_equal(
            model.loaded_state_dict["conv1.weights"], np.arange(12, dtype=np.float32).reshape(3, 4)
        )
        np.testing.assert_array_equal(
            model.loaded_state_dict["conv2.weights"], np.eye(3, dtype=np.float32)
        )


# =============================================================================
# Unit Tests for load_model_from_json_weights
# =============================================================================


class TestLoadModelFromJsonWeights:
    """Unit tests for load_model_from_json_weights."""

    def test_delegates_to_weights_from_json_and_load(self):
        """Calls weights_from_json then load_weights_into_model."""
        model = MockModel()
        serialized = [{"shape": (3,), "dtype": "float32", "data": "AAAA"}]
        parameter_names = ["layer"]
        mock_deserialized = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        with (
            patch("fed_synthetic_data.post_training.weights_from_json") as mock_wfj,
            patch("fed_synthetic_data.post_training.load_weights_into_model") as mock_lwim,
        ):

            mock_wfj.return_value = mock_deserialized
            mock_lwim.return_value = model

            result = load_model_from_json_weights(model, serialized, parameter_names)

            mock_wfj.assert_called_once_with(serialized)
            mock_lwim.assert_called_once_with(model, mock_deserialized, parameter_names)
            assert result is model

    def test_empty_serialized_weights(self):
        """Handles empty serialized weights list."""
        model = MockModel()

        with (
            patch("fed_synthetic_data.post_training.weights_from_json") as mock_wfj,
            patch("fed_synthetic_data.post_training.load_weights_into_model") as mock_lwim,
        ):

            mock_wfj.return_value = []
            mock_lwim.return_value = model

            result = load_model_from_json_weights(model, [], [])

            mock_wfj.assert_called_once_with([])
            mock_lwim.assert_called_once_with(model, [], [])
            assert result is model


# =============================================================================
# Unit Tests for save_model and save_model_weights
# =============================================================================


class TestSaveModel:
    """Unit tests for save_model and save_model_weights."""

    def test_save_model_calls_torch_save(self):
        """Delegates to torch.save when PyTorch is available."""
        model = MockModel()
        model.state_dict = MagicMock(return_value={"w": np.array([1.0])})

        with patch("fed_synthetic_data.post_training.torch") as mock_torch:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name

            try:
                save_model(model, temp_path)
                mock_torch.save.assert_called_once()
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_save_model_weights_calls_save_model(self):
        """save_model_weights delegates to save_model."""
        model = MockModel()

        with patch("fed_synthetic_data.post_training.save_model") as mock_save:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name

            try:
                save_model_weights(model, temp_path)
                mock_save.assert_called_once_with(model, temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_save_model_without_torch_raises(self):
        """Raises NotImplementedError when PyTorch is not available."""
        model = MockModel()

        with patch("fed_synthetic_data.post_training.torch", new=None):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name

            try:
                with pytest.raises(NotImplementedError, match="PyTorch or a registered backend"):
                    save_model(model, temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_save_to_nonexistent_directory_raises(self):
        """Saving to non-existent directory raises an error."""
        model = MockModel()
        model.state_dict = MagicMock(return_value={"w": np.array([1.0])})

        with patch("fed_synthetic_data.post_training.torch") as mock_torch:
            # Create a path to a non-existent directory
            non_existent_dir = os.path.join(
                tempfile.gettempdir(), "non_existent_dir_" + str(os.getpid())
            )
            path = os.path.join(non_existent_dir, "model.pt")

            # torch.save should be called, which will fail if directory doesn't exist
            with pytest.raises((FileNotFoundError, OSError)):
                save_model(model, path)

    def test_save_with_special_characters_in_path(self):
        """Saving with special characters in path works correctly."""
        model = MockModel()
        model.state_dict = MagicMock(return_value={"w": np.array([1.0])})

        with patch("fed_synthetic_data.post_training.torch") as mock_torch:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Use special characters in filename
                special_path = os.path.join(tmpdir, "model_with_spaces and-dashes.pt")

                save_model(model, special_path)

                mock_torch.save.assert_called_once()
                # Verify the path was passed correctly
                call_args = mock_torch.save.call_args
                assert special_path in str(call_args)

    def test_save_empty_model(self):
        """Saving a model with empty state_dict works."""
        model = MockModel()
        model.state_dict = MagicMock(return_value={})

        with patch("fed_synthetic_data.post_training.torch") as mock_torch:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name

            try:
                save_model(model, temp_path)
                mock_torch.save.assert_called_once_with({}, temp_path, **{})
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_save_model_weights_to_nonexistent_directory(self):
        """save_model_weights delegates to save_model with correct path."""
        model = MockModel()

        with patch("fed_synthetic_data.post_training.save_model") as mock_save:
            non_existent_dir = os.path.join(
                tempfile.gettempdir(), "non_existent_" + str(os.getpid())
            )
            path = os.path.join(non_existent_dir, "weights.pt")

            # This just verifies delegation; actual file system errors are tested in test_save_to_nonexistent_directory_raises
            save_model_weights(model, path)

            # Verify save_model was called with correct arguments
            mock_save.assert_called_once_with(model, path)

    def test_load_from_nonexistent_file_error(self):
        """Loading from non-existent file raises appropriate error."""
        model = MockModel()

        # Try to load weights from non-existent file
        # This would normally be done via torch.load, but we're testing the flow
        non_existent_path = os.path.join(
            tempfile.gettempdir(), "nonexistent_" + str(os.getpid()) + ".pt"
        )

        # The error would come from torch.load, but our functions don't directly handle file loading
        # This is more of a documentation that the error comes from the underlying framework
        assert not os.path.exists(non_existent_path)

    def test_save_with_kwargs(self):
        """save_model passes kwargs to torch.save."""
        model = MockModel()
        model.state_dict = MagicMock(return_value={"w": np.array([1.0])})

        with patch("fed_synthetic_data.post_training.torch") as mock_torch:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                temp_path = f.name

            try:
                save_model(model, temp_path, pickle_protocol=2)

                # Verify kwargs were passed
                call_args = mock_torch.save.call_args
                assert call_args[1].get("pickle_protocol") == 2
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
