"""
Utility functions for federated synthetic data generation.

This module provides general utility functions used across the library.

TODO: Implement utility functions for data manipulation, logging, and other common tasks.
"""

import base64
import math
import numpy as np
import pandas as pd


def weights_to_json(weights: list[np.ndarray] | dict[str, np.ndarray]) -> list[dict]:
    """
    Serialise model weights for JSON transport.

    Accepts either a list of arrays (unnamed) or a dict mapping parameter names
    to arrays. When a dict is supplied every entry carries a ``"name"`` field so
    that the parameter name survives the serialisation round-trip.

    Args:
        weights: Model weights to serialise. Either a ``list[np.ndarray]`` or a
            ``dict[str, np.ndarray]`` (parameter name → array). All elements
            must already be numpy arrays.

    Returns:
        list[dict]: List of dictionaries representing the weights in
            JSON-serialisable format. Dict entries include an additional
            ``"name"`` field with the parameter name.
    """
    if isinstance(weights, dict):
        return [
            {
                "name": name,
                "shape": w.shape,
                "dtype": str(w.dtype),
                "data": base64.b64encode(w.tobytes()).decode(),
            }
            for name, w in weights.items()
        ]
    return [
        {"shape": w.shape, "dtype": str(w.dtype), "data": base64.b64encode(w.tobytes()).decode()}
        for w in weights
    ]


def weights_from_json(entries: list[dict]) -> list[np.ndarray] | dict[str, np.ndarray]:
    """
    Deserialise model weights from JSON transport format.

    Each dictionary must contain:
        - "data": Base64-encoded weight data as a string.
        - "dtype": A string representing the data type of the weight.
        - "shape": A tuple (or list) representing the shape of the weight array.

    When every entry also contains a ``"name"`` field (as produced by
    :func:`weights_to_json` when called with a dict), the result is a
    ``dict[str, np.ndarray]`` keyed by parameter name, preserving the
    name → array association. When no entries carry a ``"name"`` field the
    original ``list[np.ndarray]`` is returned for backward-compatibility.

    Args:
        entries (list[dict]): List of serialised weight dictionaries.

    Returns:
        list[np.ndarray] | dict[str, np.ndarray]: Deserialised model weights.
    """
    if not entries:
        return []
    if "name" in entries[0]:
        return {
            e["name"]: np.frombuffer(base64.b64decode(e["data"]), dtype=e["dtype"]).reshape(
                e["shape"]
            )
            for e in entries
        }
    return [
        np.frombuffer(base64.b64decode(e["data"]), dtype=e["dtype"]).reshape(e["shape"])
        for e in entries
    ]


def _replace_non_finite(obj):
    """Recursively replace non-finite floats with JSON-safe sentinel strings."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return "NaN"
    elif isinstance(obj, dict):
        return {k: _replace_non_finite(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_non_finite(v) for v in obj]
    return obj


def _restore_non_finite(obj):
    """Recursively restore JSON-safe sentinel strings back to Python floats."""
    if obj == "Infinity":
        return float("inf")
    if obj == "-Infinity":
        return float("-inf")
    if obj == "NaN":
        return float("nan")
    if isinstance(obj, dict):
        return {k: _restore_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_non_finite(v) for v in obj]
    return obj


def federated_state_to_json(state: dict) -> dict:
    """
    Produce a fully JSON-serialisable representation of a ``federated_state`` dict.

    Handles the three standard keys returned by fed-mostlyai-engine:

    * ``"model_weights"`` — serialised via :func:`weights_to_json` (dict path).
    * ``"training_metrics"`` — passed through unchanged (all values are standard
      JSON types).
    * ``"lr_scheduler_state"`` — any ``float('inf')``, ``float('-inf')``, or
      ``float('nan')`` values are replaced recursively with the sentinel strings
      ``"Infinity"``, ``"-Infinity"``, and ``"NaN"`` so that ``json.dumps``
      does not raise.

    Missing keys are silently ignored so that partial ``federated_state`` dicts
    (e.g. without ``"training_metrics"``) are handled without error.

    Args:
        state (dict): A ``federated_state`` dict as returned by fed-mostlyai-engine.

    Returns:
        dict: A new dict whose values are fully JSON-serialisable.
    """
    result = {}
    if "model_weights" in state:
        result["model_weights"] = weights_to_json(state["model_weights"])
    if "training_metrics" in state:
        result["training_metrics"] = state["training_metrics"]
    if "lr_scheduler_state" in state:
        result["lr_scheduler_state"] = _replace_non_finite(state["lr_scheduler_state"])
    return result


def federated_state_from_json(state: dict) -> dict:
    """
    Invert :func:`federated_state_to_json`, recovering the original Python objects.

    * ``"model_weights"`` — deserialised via :func:`weights_from_json`.
    * ``"training_metrics"`` — passed through unchanged.
    * ``"lr_scheduler_state"`` — sentinel strings ``"Infinity"``, ``"-Infinity"``,
      and ``"NaN"`` are converted back to the corresponding Python floats.

    Missing keys are silently ignored.

    Args:
        state (dict): A JSON-deserialised ``federated_state`` dict (as produced
            by :func:`federated_state_to_json`).

    Returns:
        dict: The recovered ``federated_state`` with numpy weight arrays and
            native Python floats.
    """
    result = {}
    if "model_weights" in state:
        result["model_weights"] = weights_from_json(state["model_weights"])
    if "training_metrics" in state:
        result["training_metrics"] = state["training_metrics"]
    if "lr_scheduler_state" in state:
        result["lr_scheduler_state"] = _restore_non_finite(state["lr_scheduler_state"])
    return result


def sort_columns(df: pd.DataFrame, column_order: list[str] | None = None) -> pd.DataFrame:
    """
    Sort the columns of a DataFrame alphabetically, or reorder them according to a
    provided column order.

    Ensures a consistent column order across federated nodes. This is required
    when flexible_generation is disabled in TabularARGN, which enforces a strict
    column order at generation time.

    If no column_order is provided, the columns are sorted with numeric types first
    (sorted numerically), then string types (sorted alphabetically). This handles
    mixed-type column names safely. For duplicate column names, uses iloc-based
    reordering to avoid pandas errors.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_order (list[str] | None): An explicit column order to apply. If None,
            columns are sorted with numbers first, then strings.

    Returns:
        pd.DataFrame: A re-indexed DataFrame with columns in the specified order.

    Raises:
        ValueError: If column_order is provided but contains columns not in df.
    """
    if column_order is None:
        # Handle mixed types by sorting: numbers first (by value), then strings (alphabetically)
        columns = list(df.columns)

        def sort_key(col):
            # Numbers come first (group 0), strings come second (group 1)
            if isinstance(col, (int, float, np.integer, np.floating)):
                return (0, float(col))
            else:
                return (1, str(col))

        column_order = sorted(columns, key=sort_key)

    # Check for duplicate columns - if present, we need to use iloc-based reordering
    if len(set(column_order)) < len(column_order):
        # Has duplicates - find the permutation indices
        col_list = list(df.columns)
        # Build a mapping from column name to positions in column_order
        # For duplicates, we maintain their relative order
        order_indices = []
        for col in column_order:
            # Find the first occurrence of this column in the original df that hasn't been used yet
            for i, orig_col in enumerate(col_list):
                if orig_col == col and i not in order_indices:
                    order_indices.append(i)
                    break
        return df.iloc[:, order_indices]

    return df.reindex(column_order, axis=1)
