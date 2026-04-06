from __future__ import annotations

from collections.abc import Callable
import fnmatch
from typing import Any

import flax.traverse_util as ftu
import jax


def flat(tree):
    return {".".join(k): v for k, v in ftu.flatten_dict(tree).items()}


def unflat(tree):
    return ftu.unflatten_dict({tuple(k.split(".")): v for k, v in tree.items()})


def merge(*trees):
    out = {}
    for tree in trees:
        f = flat(tree)

        # last-wins across scalar↔subtree conflicts
        # - new key "x" wipes old "x.*"
        # - new keys "x.*" wipe old "x"
        for k in list(f.keys()):
            prefix = k + "."

            # case 1: new scalar/subtree root wipes old subtree
            out = {kk: vv for kk, vv in out.items() if not kk.startswith(prefix)}

            # case 2: new subtree wipes old scalar at any ancestor
            # e.g. k="x.y" -> remove "x"
            parts = k.split(".")
            for i in range(1, len(parts)):
                anc = ".".join(parts[:i])
                out.pop(anc, None)

        out |= f

    return unflat(out)


def drop(tree: dict, keys: list[str]) -> dict:
    return {k: v for k, v in tree.items() if k not in keys}


def drop_fn(tree: dict, fn: Callable[[str, Any], bool]) -> dict:
    """Drop leaves from a nested dict where fn(key, leaf) is True."""
    return unflat({k: v for k, v in flat(tree).items() if not fn(k, v)})


def do_fn_key(x: dict, keymatch: str, fn: Callable):
    """apply fn to keys that match keymatch. glob pattern"""
    matches = fnmatch.filter(flat(x).keys(), keymatch)
    y = {k: item for k, item in flat(x).items() if k in matches}
    if not y:
        raise KeyError(f"No keys match {keymatch}")

    return merge(x, unflat(jax.tree.map(fn, y)))
    # y = jax.tree.map(fn, y)
    # x = flat(x) | y
    # x = unflat(x)
    # return x
