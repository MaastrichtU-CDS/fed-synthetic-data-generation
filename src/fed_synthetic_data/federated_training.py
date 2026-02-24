"""
Federated training utilities for synthetic data generation.

This module provides classes and functions for coordinating federated training
of synthetic data generators across multiple nodes.
"""

import pandas as pd

from typing import Optional
from mostlyai.engine import TabularARGN

EPOCHS_PER_ITERATION = 1


def train_single_iteration(
    data: pd.DataFrame,
    model: Optional[TabularARGN] = None,
) -> TabularARGN:
    """
    Perform a single iteration of training for the given model using the provided data.

    Args:
        data (pd.DataFrame): The training data.
        model (TabularARGN): The model to train.

    Returns:
        TabularARGN: The updated model after training.
    """
    if model is None:
        model = TabularARGN(max_epochs=EPOCHS_PER_ITERATION)
    else:
        if model.max_epochs is None or model.max_epochs != EPOCHS_PER_ITERATION:
            # TODO Warn the user that the model's max_epochs is not set to the expected value
            pass

    model.fit(data)

    return model


def extract_model_weights() -> None:
    pass


def aggregate_model_updates() -> None:
    pass


def aggregation_model_weights() -> None:
    pass
