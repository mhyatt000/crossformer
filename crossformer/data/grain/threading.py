"""Helpers for configuring Google Grain threading and multiprocessing."""

from __future__ import annotations

import grain.python as gp


def create_read_options(
    *,
    num_threads: int | None = None,
    prefetch_buffer_size: int | None = None,
) -> gp.ReadOptions:
    """Constructs :class:`grain.python.ReadOptions` with partial overrides."""

    options = gp.ReadOptions()
    if num_threads is not None:
        options.num_threads = int(num_threads)
    if prefetch_buffer_size is not None:
        options.prefetch_buffer_size = int(prefetch_buffer_size)
    return options


def create_multiprocessing_options(
    *,
    num_workers: int | None = None,
    per_worker_buffer_size: int | None = None,
    enable_profiling: bool = False,
) -> gp.MultiprocessingOptions:
    """Creates :class:`grain.python.MultiprocessingOptions` with overrides."""

    options = gp.MultiprocessingOptions()
    if num_workers is not None:
        options.num_workers = int(num_workers)
    if per_worker_buffer_size is not None:
        options.per_worker_buffer_size = int(per_worker_buffer_size)
    options.enable_profiling = enable_profiling
    return options
