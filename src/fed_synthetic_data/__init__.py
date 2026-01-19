"""
fed-synthetic-data - A library for federated synthetic data generation

This library provides tools and utilities for generating synthetic data in a
federated learning context with privacy-preserving capabilities.
"""

from .synthetic_generator import (
    TabularSyntheticGenerator,
    train_federated_generator,
)

from .federated_training import (
    FederatedTrainer,
    aggregate_model_updates,
)

from .privacy_measures import (
    apply_differential_privacy,
    compute_privacy_budget,
)

from .utils import (
    safe_log,
    validate_data,
)

__all__ = [
    "TabularSyntheticGenerator",
    "train_federated_generator",
    "FederatedTrainer",
    "aggregate_model_updates",
    "apply_differential_privacy",
    "compute_privacy_budget",
    "safe_log",
    "validate_data",
]

__version__ = "0.1.0"
