from __future__ import annotations

import os
from pathlib import Path
import resource

import grain
import jax
import numpy as np
import pytest

from crossformer.data.arec.arec import ArrayRecordBuilder

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _has_2_gpus() -> bool:
    return len(jax.devices()) >= 2


LOW_FD_LIMIT = 48
MED_FD_LIMIT = 256
HIGH_FD_LIMIT = 4096


def _step(eid: int, sid: int) -> dict:
    img = np.zeros((480, 480, 3), dtype=np.uint8)
    vec = np.zeros((8,), dtype=np.float32)
    return {
        "episode_id": int(eid),
        "step_id": int(sid),
        "observation": {
            "image": {"worm": img},
            "proprio": {"single": vec.copy()},
        },
        "action": {"single": vec.copy()},
        "info": {
            "episode": np.array([eid], dtype=np.int32),
            "step": np.array([sid], dtype=np.int32),
        },
        "language_instruction": "test",
        "language_embedding": np.zeros((512,), dtype=np.float32),
        "discount": 1.0,
        "reward": 0.0,
        "is_terminal": False,
        "is_first": sid == 0,
        "is_last": sid == 0,
    }


def _build_tmp_dataset(cache_root: Path, *, name: str, shards: int) -> None:
    builder = ArrayRecordBuilder(
        name=name,
        version="0.0.1",
        branch="main",
        root=str(cache_root),
        shard_size=1,
        writer_options="group_size:1",
    )

    def stream():
        # Builder currently emits one trailing shard at this shard_size,
        # so write N-1 records to get N shard files.
        n = shards - 1 if shards > 1 else 1
        for i in range(n):
            yield _step(eid=i, sid=0)

    builder.prepare(stream)
    built_shards = sorted(builder.root.glob("data-*.arrayrecord"))
    assert len(built_shards) == shards


def _open_source(cache_root: Path, name: str):
    return ArrayRecordBuilder(name=name, version="0.0.1", branch="main", root=str(cache_root)).source


def _minimal_pipeline(source, *, read_threads: int, mp_workers: int, mp_buffer: int):
    return (
        grain.MapDataset.source(source)
        .repeat()
        .to_iter_dataset(grain.ReadOptions(num_threads=read_threads))
        .batch(1, drop_remainder=True)
        .mp_prefetch(grain.MultiprocessingOptions(num_workers=mp_workers, per_worker_buffer_size=mp_buffer))
    )


def _iterate_500_batches(ds) -> None:
    it = iter(ds)
    for _ in range(500):
        next(it)


def _apply_fd_limit(limit: int) -> tuple[int, int]:
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(limit, old_hard), old_hard))
    return old_soft, old_hard


def _restore_fd_limit(old: tuple[int, int]) -> None:
    resource.setrlimit(resource.RLIMIT_NOFILE, old)


def _run_case(
    tmp_path: Path,
    *,
    shards: int,
    fd_limit: int,
    read_threads: int,
    mp_workers: int,
    mp_buffer: int,
) -> None:
    cache_root = tmp_path / ".cache" / "arrayrecords"
    dname = f"ds_{shards}"
    _build_tmp_dataset(cache_root, name=dname, shards=shards)

    old = _apply_fd_limit(fd_limit)
    try:
        src = _open_source(cache_root, dname)
        ds = _minimal_pipeline(
            src,
            read_threads=read_threads,
            mp_workers=mp_workers,
            mp_buffer=mp_buffer,
        )
        _iterate_500_batches(ds)
    finally:
        _restore_fd_limit(old)


@pytest.mark.skipif(os.name != "posix", reason="FD-limit behavior requires POSIX rlimit")
@pytest.mark.skipif(not _has_2_gpus(), reason="requires 2+ GPUs")
@pytest.mark.parametrize("read_threads", [1, 4, 16], ids=lambda x: f"thread{x}")
@pytest.mark.parametrize("mp_workers", [1, 2, 4, 8], ids=lambda x: f"proc{x}")
@pytest.mark.parametrize("mp_buffer", [1, 8], ids=lambda x: f"buf{x}")
@pytest.mark.parametrize("shards", [3, 40], ids=lambda x: f"{x}shard")
@pytest.mark.parametrize("fd_limit", [LOW_FD_LIMIT, MED_FD_LIMIT, HIGH_FD_LIMIT], ids=lambda x: f"fd{x}")
def test_fd_limit_matrix(
    tmp_path,
    monkeypatch,
    shards: int,
    fd_limit: int,
    read_threads: int,
    mp_workers: int,
    mp_buffer: int,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    _run_case(
        tmp_path,
        shards=shards,
        fd_limit=fd_limit,
        read_threads=read_threads,
        mp_workers=mp_workers,
        mp_buffer=mp_buffer,
    )
