"""Runtime configuration guide for Grain I/O, echoing :mod:`tests.grain.test_runtime_options`."""

from __future__ import annotations

import grain.python as gp

from crossformer.data.grain import sharding
from crossformer.data.grain import threading


def shard_options_demo() -> tuple[gp.ShardOptions, gp.ShardOptions, gp.ShardByJaxProcess]:
    """Show default, parameterized, and JAX-aware sharding options."""
    default = sharding.create_shard_options()
    manual = sharding.create_shard_options(shard_count=4, shard_index=2, drop_remainder=True)
    jax_process = sharding.create_shard_options(use_jax_process=True)
    return default, manual, jax_process


def read_options_demo() -> gp.ReadOptions:
    """Override reader parallelism and buffering."""
    return threading.create_read_options(num_threads=8, prefetch_buffer_size=16)


def multiprocessing_options_demo() -> gp.MultiprocessingOptions:
    """Tune worker pool behaviour for multiprocess readers."""
    return threading.create_multiprocessing_options(
        num_workers=4,
        per_worker_buffer_size=32,
        enable_profiling=True,
    )


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    default, manual, jax_process = shard_options_demo()
    print("default shards", default.shard_count, default.shard_index)
    print("manual shards", manual.shard_count, manual.shard_index, manual.drop_remainder)
    print("jax process", type(jax_process).__name__)
    print("read options", read_options_demo().num_threads)
    print("multiprocessing", multiprocessing_options_demo().num_workers)
