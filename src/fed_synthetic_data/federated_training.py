"""
Federated training utilities for synthetic data generation.

This module provides helper functions for coordinating federated training
of synthetic data generators across multiple nodes. These are intended to be
imported into a vantage6 algorithm.
"""

import base64
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


def weights_to_json(weights: list[np.ndarray]) -> list[dict]:
    """
    Serialise model weights for JSON transport.

    Args:
        weights (list[np.ndarray]): List of model weights to serialise.

    Returns:
        list[dict]: List of dictionaries representing the weights in JSON-serialisable format.
    """
    return [
        {"shape": w.shape, "dtype": str(w.dtype), "data": base64.b64encode(w.tobytes()).decode()}
        for w in weights
    ]


def weights_from_json(entries: list[dict]) -> list[np.ndarray]:
    """
    Deserialise model weights from JSON transport format.

    Each dictionary must contain:
        - "data": Base64-encoded weight data as a string.
        - "dtype": A string representing the data type of the weight.
        - "shape": A tuple (or list) representing the shape of the weight array.

    Args:
        entries (list[dict]): List of serialised weight dictionaries.

    Returns:
        list[np.ndarray]: Deserialised model weights as numpy arrays.
    """
    return [
        np.frombuffer(base64.b64decode(e["data"]), dtype=e["dtype"]).reshape(e["shape"])
        for e in entries
    ]


def evaluate_loss() -> None:
    """
    TODO implement loss evaluation logic
    :return:
    """
    pass
