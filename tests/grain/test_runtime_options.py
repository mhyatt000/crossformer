import grain.python as gp
import pytest

from crossformer.data.grain import sharding
from crossformer.data.grain import threading


def test_create_shard_options_defaults():
    options = sharding.create_shard_options()
    assert isinstance(options, gp.ShardOptions)
    assert options.shard_index == 0
    assert options.shard_count == 1


def test_create_shard_options_with_parameters():
    options = sharding.create_shard_options(
        shard_count=4, shard_index=2, drop_remainder=True
    )
    assert options.shard_index == 2
    assert options.shard_count == 4
    assert options.drop_remainder is True


def test_create_shard_options_jax_process():
    options = sharding.create_shard_options(use_jax_process=True)
    assert isinstance(options, gp.ShardByJaxProcess)


def test_create_shard_options_missing_index():
    with pytest.raises(ValueError):
        sharding.create_shard_options(shard_count=2)


def test_create_read_options_overrides():
    options = threading.create_read_options(num_threads=8, prefetch_buffer_size=16)
    assert isinstance(options, gp.ReadOptions)
    assert options.num_threads == 8
    assert options.prefetch_buffer_size == 16


def test_create_multiprocessing_options_overrides():
    options = threading.create_multiprocessing_options(
        num_workers=4,
        per_worker_buffer_size=32,
        enable_profiling=True,
    )
    assert isinstance(options, gp.MultiprocessingOptions)
    assert options.num_workers == 4
    assert options.per_worker_buffer_size == 32
    assert options.enable_profiling is True
