"""
fed-synthetic-data - A library for federated synthetic data generation

This library provides tools and utilities for generating synthetic data in a
federated learning context with privacy-preserving capabilities.
"""

from .federated_training import (
    aggregation_model_weights_weighted_average,
    evaluate_loss,
    lr_determination,
    should_stop_early,
)

from .post_training import (
    load_weights_into_model,
    load_model_from_json_weights,
    save_model,
    save_model_weights,
)

from .utils import (
    weights_to_json,
    weights_from_json,
    sort_columns,
    federated_state_to_json,
    federated_state_from_json,
)

__all__ = [
    "aggregation_model_weights_weighted_average",
    "evaluate_loss",
    "lr_determination",
    "should_stop_early",
    "load_weights_into_model",
    "load_model_from_json_weights",
    "save_model",
    "save_model_weights",
    "weights_to_json",
    "weights_from_json",
    "sort_columns",
    "federated_state_to_json",
    "federated_state_from_json",
]

__version__ = "0.1.0"
