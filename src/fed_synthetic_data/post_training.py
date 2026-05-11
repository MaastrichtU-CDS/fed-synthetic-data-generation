"""
Post federated training utilities for synthetic data generation.

This module provides functions for loading JSON-serialised aggregated weights
into a model and saving the result. Reuses serialisation utilities from
utils.py.
"""

from fed_synthetic_data.utils import weights_from_json
from typing import Any, Protocol


class ModelProtocol(Protocol):
    """Minimal interface for model-agnostic weight loading."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...


def load_weights_into_model(
    model: ModelProtocol,
    deserialised_weights: list[Any],
    parameter_names: list[str],
) -> ModelProtocol:
    """
    Load deserialised weights into a model.

    Args:
        model: Model instance (must support load_state_dict/state_dict).
        deserialised_weights: Weights as numpy arrays (from weights_from_json).
        parameter_names: Names matching model.state_dict() keys.

    Returns:
        Model with loaded weights.

    Raises:
        ValueError: If length or shape mismatch.
    """
    if len(deserialised_weights) != len(parameter_names):
        raise ValueError(
            f"Mismatch: {len(deserialised_weights)} weights vs {len(parameter_names)} names"
        )

    state_dict = {name: weight for name, weight in zip(parameter_names, deserialised_weights)}
    model.load_state_dict(state_dict)
    return model


def load_model_from_json_weights(
    model: ModelProtocol,
    serialized_weights: list[dict],
    parameter_names: list[str],
) -> ModelProtocol:
    """
    Load JSON-serialised weights into a model.

    Reuses weights_from_json from utils.py for deserialisation.

    Args:
        model: Model instance to load weights into.
        serialized_weights: JSON-serialisable weight entries (from federation layer).
        parameter_names: Names matching model.state_dict() keys.

    Returns:
        Model with loaded aggregated weights.
    """
    deserialised_weights = weights_from_json(serialized_weights)
    return load_weights_into_model(model, deserialised_weights, parameter_names)


def save_model(model: ModelProtocol, path: str, **kwargs) -> None:
    """
    Save a model with loaded weights to disk.

    Delegates to framework-specific save logic. For PyTorch models,
    saves state_dict by default.

    Args:
        model: Model to save.
        path: Destination path.
        **kwargs: Framework-specific options (e.g., save_format for PyTorch).
    """
    try:
        import torch

        torch.save(model.state_dict(), path, **kwargs)
    except ImportError:
        raise NotImplementedError("Model saving requires PyTorch or a registered backend")


def save_model_weights(model: ModelProtocol, path: str) -> None:
    """
    Save only the model weights (not the full model) to disk.

    For PyTorch: saves state_dict. Can be reloaded with load_weights_into_model.

    Args:
        model: Model with loaded weights.
        path: Destination path.
    """
    save_model(model, path)
