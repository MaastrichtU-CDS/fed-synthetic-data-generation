"""
Federated training utilities for synthetic data generation.

This module provides helper functions for coordinating federated training
of synthetic data generators across multiple nodes. These are intended to be
imported into a vantage6 algorithm.
"""

import numpy as np

from functools import reduce
from typing import Any, cast

def aggregation_model_weights_weighted_average(
    results: list[tuple[list[np.ndarray] | dict[str, np.ndarray], int]],
) -> list[np.ndarray] | dict[str, np.ndarray]:
    """
    Compute weighted average of model parameters.

    Adapted from Flower (Apache-2.0):
    https://github.com/flwrlabs/flower/blob/983b0f29/framework/py/flwr/server/strategy/aggregate.py#L28-L43

    Weights must be numpy arrays. Use :func:`fed_synthetic_data.utils.weights_from_json`
    to deserialise JSON-transported weights before calling this function.

    Accepts either the list form ``list[tuple[list[np.ndarray], int]]`` (positional,
    existing behaviour) or the dict form ``list[tuple[dict[str, np.ndarray], int]]``
    (keyed by parameter name). When the dict form is used, parameters are matched by
    name so differing key orders between nodes are handled correctly. A
    :exc:`ValueError` is raised if the sets of parameter names differ between any
    two nodes.

    Args:
        results (list[tuple[list[np.ndarray] | dict[str, np.ndarray], int]]):
            List of tuples containing model weights (as a list or dict of numpy
            arrays) and the number of training examples for each node.

    Returns:
        list[np.ndarray] | dict[str, np.ndarray]: The aggregated model weights,
            in the same format as the input weights (list or dict).
            An empty list is returned when results is empty.

    Raises:
        ValueError: If any sample count is negative, if total samples is not
            positive, or (for dict inputs) if the parameter-name sets differ
            between nodes.
    """
    if not results:
        return []

    for _, n in results:
        if n < 0:
            raise ValueError(f"Sample count must be non-negative, got {n}")

    num_examples_total = sum(n for (_, n) in results)

    if num_examples_total <= 0:
        raise ValueError(f"Total number of samples must be positive, got {num_examples_total}")

    first_weights = results[0][0]
    if isinstance(first_weights, dict):
        dict_results: list[tuple[dict[str, np.ndarray], int]] = [
            (cast(dict[str, np.ndarray], w), n) for w, n in results
        ]
        reference_keys = list(first_weights.keys())
        reference_key_set = set(reference_keys)
        for weights, _ in dict_results[1:]:
            if set(weights.keys()) != reference_key_set:
                raise ValueError(
                    f"Parameter name mismatch between nodes: "
                    f"expected {reference_key_set}, got {set(weights.keys())}"
                )
        return {
            key: reduce(np.add, [weights[key] * n for weights, n in dict_results])
            / num_examples_total
            for key in reference_keys
        }

    weighted_weights = [[layer * n for layer in weights] for weights, n in results]
    return [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights, strict=True)
    ]


def evaluate_loss(loss_results: list[dict]) -> Any:
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


def lr_determination():
    """
    TODO: Determine learning rate for the next round of federated training.
    This is to ensure that the LR can be coordinated across all nodes.
    """
    pass
