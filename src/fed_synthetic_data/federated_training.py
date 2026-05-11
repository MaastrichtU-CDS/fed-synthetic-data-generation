"""
Federated training utilities for synthetic data generation.

This module provides helper functions for coordinating federated training
of synthetic data generators across multiple nodes. These are intended to be
imported into a vantage6 algorithm.
"""

import numpy as np

from functools import reduce


def aggregation_model_weights_weighted_average(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """
    Compute weighted average of model parameters.

    Ported from Flower (Apache-2.0):
    https://github.com/flwrlabs/flower/blob/983b0f29/framework/py/flwr/server/strategy/aggregate.py#L28-L43

    Args:
        results (list[tuple[list[np.ndarray], int]]): List of tuples containing model weights
            and the number of training examples for each node.

    Returns:
        list[np.ndarray]: The aggregated model weights.
    """
    num_examples_total = sum(n for (_, n) in results)
    weighted_weights = [
        [layer * n for layer in weights]
        for weights, n in results
    ]
    return [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights, strict=True)
    ]


def evaluate_loss(loss_results: list[dict]) -> float:
    """
    Compute the weighted average loss across federated sites.

    Each site reports its loss together with the number of samples it was evaluated on.
    The aggregate loss is the sample-weighted average, which matches how the model weights are aggregated.

    Args:
        loss_results (list[dict]): Per-site results, each containing:
            - "loss": float - The loss value for this site.
            - "samples": int - Number of samples evaluated at this site.

    Returns:
        float: The sample-weighted average loss across all sites.

    Raises:
        ValueError: If 'loss_results' is empty or if the 'total samples' is zero.
    """
    if not loss_results:
        raise ValueError("loss_results cannot be empty")

    total_samples = sum(r["samples"] for r in loss_results)
    if total_samples == 0:
        raise ValueError("total number of samples must be greater than zero")

    return sum(r["loss"] * r["samples"] for r in loss_results) / total_samples


def should_stop_early(
        loss_history: list[float],
        patience: int = 5,
        min_delta: float = 1e-4,
) -> bool:
    """
    Decide whether federated training should stop early based on loss history.

    Returns True when the best loss seen has not improved by at least 'min_delta' for the last 'patience' iterations.
    The caller is responsible for maintaining the history
    (e.g. by appending the result of :func: 'evaluate_loss' after each round).

    Args:
        loss_history (list[float]): Aggregate loss per iteration, in order.
        patience (int): Number of iterations without improvement to tolerate
            before stopping. Defaults to 5.
        min_delta (float): Minimum decrease in loss that counts as an
            improvement. Defaults to 1e-4.

    Returns:
        bool: True if early stopping should be triggered, False otherwise.
    """
    if len(loss_history) <= patience:
        return False

    best_before_window = min(loss_history[:-patience])
    recent_best = min(loss_history[-patience:])
    return recent_best > best_before_window - min_delta
