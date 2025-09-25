"""Utility helpers overview inspired by :mod:`tests.grain.test_utils_module`."""

from __future__ import annotations

import numpy as np

from crossformer.data.grain import utils


def tree_map_demo() -> dict:
    """Apply a lambda across nested dictionaries without mutation."""
    tree = {"a": 1, "b": {"c": 2, "d": 3}}
    result = utils.tree_map(lambda x: x * 2, tree)
    return {"result": result, "original": tree}


def tree_merge_demo() -> dict:
    """Merge dictionaries preferring override values."""
    base = {"a": 1, "nested": {"left": 5, "shared": {"x": 1}}}
    override = {"nested": {"right": 6, "shared": {"y": 2}}, "extra": 3}
    return utils.tree_merge(base, override)


def clone_structure_demo() -> dict:
    """Clone nested structures without sharing array references."""
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    original = {"x": array, "nested": {"y": [1, 2, 3]}}
    clone = utils.clone_structure(original)
    clone["nested"]["y"].append(4)
    return {"clone": clone, "original": original}


def is_padding_demo() -> dict[str, np.ndarray]:
    """Inspect padding detection for mixed data types."""
    cases = {
        "float": np.zeros((2, 2), dtype=np.float32),
        "string": np.array(["", "foo"], dtype="<U3"),
        "bool": np.array([True, False], dtype=bool),
        "nested": {"a": np.zeros((2,), dtype=np.float32), "b": np.array(["", ""], dtype="<U1")},
    }
    return {name: utils.is_padding(value) for name, value in cases.items()}


def to_padding_demo() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert values into padding tokens that mirror original dtypes."""
    return (
        utils.to_padding(np.array([1, 2], dtype=np.int32)),
        utils.to_padding(np.array(["a"], dtype="U1")),
        utils.to_padding(np.array([True, False], dtype=bool)),
    )


def ensure_numpy_demo() -> tuple[np.ndarray, dict]:
    """Convert sequences to NumPy arrays and wrap mappings safely."""
    array = utils.ensure_numpy([1, 2, 3])
    mapping = utils.as_dict({"a": 1})
    mapping["b"] = 2
    empty = utils.as_dict(None)
    return array, mapping | empty


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print("tree map", tree_map_demo())
    print("tree merge", tree_merge_demo())
    print("clone", clone_structure_demo()["clone"]["nested"]["y"])
    print("padding keys", list(is_padding_demo().keys()))
    print("to padding dtypes", [arr.dtype for arr in to_padding_demo()])
    print("ensure numpy", ensure_numpy_demo()[0].dtype)
