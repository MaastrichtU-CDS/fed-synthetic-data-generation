"""
Federated training utilities for synthetic data generation.

This module provides helper functions for coordinating federated training
of synthetic data generators across multiple nodes. These are intended to be
imported into a vantage6 algorithm.
"""

import math
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


def reducelr_on_plateau_step(
    current_lr: float,
    metric: float,
    state: dict[str, Any],
    *,
    mode: str = "min",
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
    threshold_mode: str = "rel",
    min_lr: float = 0.0,
    eps: float = 1e-8,
) -> tuple[float, dict[str, Any]]:
    """
    Compute the next learning rate using ReduceLROnPlateau logic.

    This is a pure function that implements the learning rate reduction logic
    from PyTorch's ReduceLROnPlateau scheduler
    (https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
    It reduces the learning rate by a factor when a metric has stopped improving
    for a specified number of iterations (patience).

    The caller is responsible for maintaining the state dictionary between
    calls. Initial state can be obtained by calling this function with an
    empty dict or None (handled internally).

    Args:
        current_lr (float): The current learning rate.
        metric (float): The current value of the monitored metric (e.g. loss).
        state (dict[str, Any]): Dictionary containing scheduler state:
            - "best": float - Best metric value seen so far.
            - "num_bad_epochs": int - Number of consecutive epochs without
              improvement.
        mode (str): One of "min" or "max". In "min" mode, the learning rate
            will be reduced when the metric stops decreasing; in "max" mode,
            when the metric stops increasing. Defaults to "min".
        factor (float): Factor by which the learning rate will be reduced:
            new_lr = lr * factor. Must be < 1.0. Defaults to 0.1.
        patience (int): Number of epochs with no improvement after which
            the learning rate will be reduced. Defaults to 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Defaults to 1e-4.
        threshold_mode (str): One of "rel" or "abs". In "rel" mode, the
            dynamic threshold is computed relative to the best value;
            in "abs" mode, it is an absolute difference. Defaults to "rel".
        min_lr (float): A lower bound on the learning rate. Defaults to 0.0.
        eps (float): Minimal decay applied to lr. If the difference between
            new and old lr is smaller than eps, the update is ignored.
            Defaults to 1e-8.

    Returns:
        tuple[float, dict[str, Any]]: A tuple containing:
            - The new learning rate.
            - The updated state dictionary.

    Raises:
        ValueError: If mode is not "min" or "max".
        ValueError: If threshold_mode is not "rel" or "abs".
        ValueError: If factor is >= 1.0.
    """
    if factor >= 1.0:
        raise ValueError("Factor should be < 1.0.")

    if mode not in {"min", "max"}:
        raise ValueError(f"mode '{mode}' is unknown! Expected 'min' or 'max'.")

    if threshold_mode not in {"rel", "abs"}:
        raise ValueError(
            f"threshold_mode '{threshold_mode}' is unknown! Expected 'rel' or 'abs'."
        )

    # Initialise state if empty
    if not state:
        state = {
            "best": math.inf if mode == "min" else -math.inf,
            "num_bad_epochs": 0,
        }

    best = state["best"]
    num_bad_epochs = state["num_bad_epochs"]

    # Check if current metric is better than best
    if mode == "min" and threshold_mode == "rel":
        is_better = metric < best * (1.0 - threshold)
    elif mode == "min" and threshold_mode == "abs":
        is_better = metric < best - threshold
    elif mode == "max" and threshold_mode == "rel":
        is_better = metric > best * (1.0 + threshold)
    else:  # mode == "max" and threshold_mode == "abs"
        is_better = metric > best + threshold

    if is_better:
        best = metric
        num_bad_epochs = 0
    else:
        num_bad_epochs += 1

    if num_bad_epochs > patience:
        new_lr = max(current_lr * factor, min_lr)
        if current_lr - new_lr > eps:
            current_lr = new_lr
            num_bad_epochs = 0
        else:
            # Update not applied due to eps, but still reset counters
            num_bad_epochs = 0

    new_state = {
        "best": best,
        "num_bad_epochs": num_bad_epochs,
    }

    return current_lr, new_state


def lr_determination(
    current_lr: float,
    metric: float,
    scheduler: str = "reducelr_on_plateau",
    state: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    """
    Determine learning rate for the next round of federated training.

    This function provides a flexible interface for computing the learning
    rate using different scheduler strategies. It ensures that the learning
    rate can be coordinated centrally across all nodes in a federated setting.

    Args:
        current_lr (float): The current learning rate.
        metric (float): The current value of the monitored metric
            (e.g. validation loss).
        scheduler (str): Name of the scheduler strategy to use.
            Currently supported: "reducelr_on_plateau". Defaults to
            "reducelr_on_plateau".
        state (dict[str, Any] | None): Dictionary containing scheduler state
            from the previous call. If None, initial state will be created.
            Defaults to None.
        **kwargs: Additional keyword arguments specific to the chosen scheduler.
            For "reducelr_on_plateau": mode, factor, patience, threshold,
            threshold_mode, min_lr, eps.

    Returns:
        tuple[float, dict[str, Any]]: A tuple containing:
            - The new learning rate for the next round.
            - The updated state dictionary to pass to the next call.

    Raises:
        ValueError: If the specified scheduler is not supported.
    """
    if state is None:
        state = {}

    if scheduler == "reducelr_on_plateau":
        return reducelr_on_plateau_step(current_lr, metric, state, **kwargs)

    raise ValueError(f"Scheduler '{scheduler}' is not supported.")
