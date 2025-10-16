"""Utility helpers for the Grain data pipeline.

The TensorFlow based pipeline under :mod:`crossformer.data.dataset` relies on
TensorFlow primitives for a large collection of small utilities such as
recursive tree mapping, padding helpers, or dictionary merges.  The Grain based
pipeline needs the same functionality but implemented using NumPy/JAX friendly
operations so that it remains completely framework agnostic.

Only a very small subset of helpers are required for the initial Grain
implementation and they are intentionally kept lightweight.  They operate on
standard Python containers (``dict``/``list``) and NumPy/JAX arrays so they can
be freely reused both inside tests and production code without pulling in
TensorFlow as a dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from typing import Any

import jax
import numpy as np

Tree = Mapping[str, Any] | dict[str, Any]


def tree_map(fn: Callable[[Any], Any], tree: Tree) -> Tree:
    """Applies ``fn`` recursively to every leaf in ``tree``.

    ``tree`` is expected to be made of nested ``dict`` instances.  Lists and
    tuples are treated as leaves to avoid surprising conversions between data
    structures.  This mirrors the behaviour of ``tf.nest.map_structure`` which
    the TensorFlow pipeline relies on.
    """

    def _map(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _map(sub_value) for key, sub_value in value.items()}
        return fn(value)

    return _map(tree)


def tree_merge(*trees: Tree) -> Tree:
    """Merges ``trees`` recursively with right-most values taking precedence."""

    if not trees:
        return {}
    result: dict[str, Any] = {}
    for tree in trees:
        for key, value in tree.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = tree_merge(result[key], value)  # type: ignore[arg-type]
            elif isinstance(value, dict):
                result[key] = tree_merge(value)
            else:
                result[key] = value
    return result


def clone_structure(value: Any) -> Any:
    """Returns a deep copy of ``value`` preserving NumPy arrays.

    ``copy.deepcopy`` would normally work but it has two undesirable
    properties:

    * it recursively copies NumPy arrays which is unnecessary and expensive
      for large tensors;
    * it is not guaranteed to work with objects backed by shared memory which
      the Grain data sources may rely on.

    This helper performs a deep copy for Python containers while keeping NumPy
    arrays (and other objects that expose ``__array__``) as views.
    """

    def _clone(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        return copy.deepcopy(value)

    return jax.tree.map(_clone, value)


def is_padding(value: Any) -> np.ndarray:
    """Returns a boolean mask indicating which entries correspond to padding.

    The heuristic mirrors the TensorFlow implementation:

    * numerical arrays are considered padding if every element is ``0``;
    * byte/Unicode arrays are padding if they are empty strings;
    * boolean arrays use ``False`` as the padding sentinel;
    * nested dictionaries are processed recursively with logical ``and``.

    The return type is always a boolean NumPy array matching the input shape.
    """

    if isinstance(value, dict):
        masks = [is_padding(sub_value) for sub_value in value.values()]
        if not masks:
            raise ValueError("Cannot infer padding mask for empty dict.")
        mask = masks[0]
        for sub_mask in masks[1:]:
            mask = np.logical_and(mask, sub_mask)
        return mask
    value = np.asarray(value)
    if value.dtype.kind in {"U", "S"}:
        return value == ""
    if value.dtype == np.bool_:
        return ~value
    return value == 0


def to_padding(value: Any) -> Any:
    """Creates a padding value with the same shape/dtype as ``value``."""

    value = np.asarray(value)
    if value.dtype.kind in {"U", "S"}:
        return np.zeros_like(value, dtype=value.dtype)
    if value.dtype == np.bool_:
        return np.zeros_like(value, dtype=bool)
    return np.zeros_like(value)


def ensure_numpy(value: Any) -> np.ndarray:
    """Converts ``value`` to a NumPy array if possible."""

    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return np.array(value)


def as_dict(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Returns ``data`` as a mutable dictionary."""

    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    return dict(data)
