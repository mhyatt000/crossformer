from __future__ import annotations

from collections.abc import Callable
import logging
import os
from typing import Any, Sequence

import jax
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
import numpy as np

from crossformer.utils.mytyping import PyTree


def _resolve_device(device):
    return device() if callable(device) else device


def compose(*fns: Callable[[PyTree], PyTree]) -> Callable[[PyTree], PyTree]:
    """Chain multiple PyTree transforms into a single callable."""

    def run(x: PyTree) -> PyTree:
        for fn in fns:
            x = fn(x)
        return x

    return run


def viz(tree: PyTree):
    jax.debug.visualize_array_sharding(tree)


def str2np(s: str, length: int | None = None) -> np.ndarray:
    """Encode a string as uint8 bytes, optionally zero-padded to a fixed length."""
    raw = np.frombuffer(s.encode("utf-8"), dtype=np.uint8)
    if length is not None:
        if len(raw) > length:
            raise ValueError(f"encoded string length {len(raw)} exceeds fixed length {length}")
        padded = np.zeros(length, dtype=np.uint8)
        padded[: len(raw)] = raw
        raw = padded
    return raw


def npstr2np(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind != "U":
        raise TypeError("expected Unicode array")
    encoded = [s.encode("utf-8") for s in arr]
    maxlen = max(len(e) for e in encoded)
    return np.array([np.frombuffer(e.ljust(maxlen, b"\0"), dtype=np.uint8) for e in encoded])


def str2jax(s: str, device=None, length: int | None = None) -> jax.Array:
    """Encode a string as uint8 bytes, optionally zero-padded to a fixed length."""
    return jnp.array(str2np(s, length=length), device=_resolve_device(device) or cpu())


def npstr2jax(arr: np.ndarray, device=None) -> jax.Array:
    return jnp.array(npstr2np(arr), device=_resolve_device(device) or cpu())


def jax2str(x: jax.Array) -> str:
    """Convert a JAX array back to a string by decoding each Unicode code point."""
    return np.array(x.astype(jnp.uint8)).tobytes().decode("utf-8")


def jpad_str(x: jax.Array, length: int) -> str:
    """Pad a jax uint string to a fixed length with 0s."""
    x = jnp.array(x.astype(jnp.uint8))
    shape = (*x.shape[:-1], length)
    padded = jnp.zeros(shape, dtype=jnp.uint8)
    return padded + x


def host_broadcast_str(x: str) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    multihost_utils.assert_equal(len(x), f"String lengths are not equal: got {len(x)} for {jax.process_index()}")
    encoded = np.array([ord(c) for c in x], dtype=np.uint8)
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    return "".join([chr(u) for u in encoded])


def shard_along_axis(x: Any, devices: Sequence[jax.Device], axis: int = 0) -> jax.Array:
    """Shard a PyTree of arrays along a given axis, putting them on device in
    the process. Works in multi-host setting as long as PyTrees are equal on all
    hosts."""
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )
    x = jax.tree.map(jnp.array, x)
    return jax.tree.map(
        lambda arr: jax.make_array_from_callback(arr.shape, sharding, lambda index: arr[index]),
        x,
    )


def merge_along_axis(x: Any, axis: int = 0) -> jax.Array:
    """Convert a PyTree of host-local arrays to a global array, concatenating and sharding along
    `axis`."""
    return multihost_utils.host_local_array_to_global_array(
        x,
        jax.sharding.Mesh(jax.devices(), "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )


def split_along_axis(x: Any, axis: int = 0) -> jax.Array:
    """Convert a PyTree of global arrays to a host-local array, splitting along `axis`."""
    return multihost_utils.global_array_to_host_local_array(
        x,
        jax.sharding.Mesh(jax.devices(), "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )


def replicate(x: Any, devices: Sequence[jax.Device] | None = None) -> jax.Array:
    """Replicate a PyTree of arrays across devices. Works in multi-host setting
    as long as PyTrees are equal on all hosts."""
    if devices is None:
        devices = jax.devices()
    sharding = jax.sharding.PositionalSharding(devices).replicate()
    x = jax.tree.map(jnp.array, x)
    return jax.tree.map(
        lambda arr: jax.make_array_from_callback(arr.shape, sharding, lambda index: arr[index]),
        x,
    )


def initialize_compilation_cache(
    cache_dir=os.path.expanduser("~/.jax_compilation_cache"),
):
    """Initializes the Jax persistent compilation cache."""
    compilation_cache.set_cache_dir(cache_dir)
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.addFilter(lambda record: "Not writing persistent cache entry for" not in record.getMessage())
