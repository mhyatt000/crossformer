"""Utility functions for configuring sharding of Grain data sources."""

from __future__ import annotations

import grain.python as gp


def create_shard_options(
    *,
    shard_count: int | None = None,
    shard_index: int | None = None,
    drop_remainder: bool = False,
    use_jax_process: bool = False,
) -> gp.ShardOptions:
    """Returns :class:`grain.python.ShardOptions` according to the strategy."""

    if use_jax_process:
        return gp.ShardByJaxProcess(drop_remainder=drop_remainder)

    if shard_count is None:
        return gp.ShardOptions(shard_index=0, shard_count=1, drop_remainder=drop_remainder)
    if shard_index is None:
        raise ValueError("shard_index must be provided when shard_count is set")
    return gp.ShardOptions(
        shard_index=int(shard_index),
        shard_count=int(shard_count),
        drop_remainder=drop_remainder,
    )
