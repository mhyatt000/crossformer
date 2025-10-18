"""Dataset helpers for working with ArrayRecord sources."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Generic, Iterable, Sequence, TypeVar

from array_record.python.array_record_data_source import ArrayRecordDataSource
import jax
import numpy as np

from crossformer.data.grain.arec.arec import unpack_record

log = logging.getLogger(__name__)


class _DecodedArrayRecord:
    """Decode ArrayRecord shards into Python dictionaries."""

    def __init__(self, shards: Iterable[Path]):
        self._shards = tuple(sorted(Path(p) for p in shards))
        self._ds = ArrayRecordDataSource([str(p) for p in self._shards])

    @property
    def shards(self) -> tuple[Path, ...]:
        return self._shards

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self._ds)

    def __getitem__(self, index: int):  # pragma: no cover - simple delegation
        return unpack_record(self._ds[index])

    def __getitems__(self, indices: Sequence[int]) -> list[dict]:
        return [unpack_record(x) for x in self._ds.__getitems__(indices)]


class _EpisodeDataset(Sequence[dict]):
    """Group step-wise ArrayRecord data into episodes."""

    def __init__(
        self,
        records: _DecodedArrayRecord,
        *,
        cache: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self._records = records
        self._cache_enabled = cache
        self._cache_dir = self._resolve_cache_dir(cache_dir)
        self._episode_indices = self._group_indices()

    def _group_indices(self) -> list[list[int]]:
        cached = self._load_cached_indices()
        log.info(f"Cached idxs: {type(cached)}")
        if cached is not None:
            return cached
        log.info("No cached episode indices found; grouping from scratch.")

        by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for idx in range(len(self._records)):
            record = self._records[idx]
            episode_id = int(record.get("episode_id", 0))
            step_id = int(record.get("step_id", idx))
            by_episode[episode_id].append((step_id, idx))

        grouped: list[list[int]] = []
        for _, pairs in sorted(by_episode.items()):
            pairs.sort(key=lambda pair: pair[0])
            grouped.append([idx for _, idx in pairs])

        self._store_cached_indices(grouped)
        return grouped

    def _resolve_cache_dir(self, cache_dir: Path | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        return root / "arrayrecords"

    def _cache_key(self) -> str | None:
        shards = getattr(self._records, "shards", None)
        if not shards:
            return None

        fingerprint_parts = []
        for shard in shards:
            try:
                stat = shard.stat()
            except OSError:
                return None
            fingerprint_parts.append(f"{shard.resolve()}::{int(stat.st_mtime)}::{stat.st_size}")

        digest = hashlib.sha1("\n".join(fingerprint_parts).encode()).hexdigest()
        return digest

    def _cache_path(self) -> Path | None:
        if not self._cache_enabled:
            return None
        key = self._cache_key()
        if key is None:
            return None
        return self._cache_dir / key / "episode_indices.json"

    def _load_cached_indices(self) -> list[list[int]] | None:
        path = self._cache_path()
        if path is None or not path.exists():
            return None
        log.info(f"Loading cached episode indices from {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return [[int(idx) for idx in group] for group in payload]

    def _store_cached_indices(self, indices: list[list[int]]) -> None:
        path = self._cache_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(indices, handle)
        except OSError:
            return

    def __len__(self) -> int:
        return len(self._episode_indices)

    def lengths(self) -> list[int]:
        """Return the lengths of all records in the dataset."""
        return [len(idxs) for idxs in self._episode_indices]

    def __getitem__(self, index: int) -> dict:
        indices = self._episode_indices[index]
        # steps = [self._records[i] for i in indices] # too slow
        steps = self._records.__getitems__(indices)
        return jax.tree.map(lambda *x: np.stack([*x]), *steps)


T = TypeVar("T")


class CacheIter(Iterator[T], Generic[T]):
    """
    Wraps a parent iterator/sequence.
    - __getitem__(i): returns parent[i], caching the result.
    - __next__(): returns next(parent), caching by increasing index (0,1,2,...).
    - __iter__(): standard iterator protocol (returns self so caching applies during iteration).
    """

    def __init__(self, parent):
        self._parent = parent
        self._cache: dict[int, T] = {}
        self._i = 0  # next index to assign for __next__-driven iteration

    def __getitem__(self, idx: int) -> T:
        if idx in self._cache:
            return self._cache[idx]
        # Prefer direct delegation if the parent supports random access
        if hasattr(self._parent, "__getitem__"):
            val = self._parent[idx]
            self._cache[idx] = val
            return val
        # Fallback: advance the parent until we reach idx
        while self._i <= idx:
            _ = next(self)  # uses our __next__, which caches
        return self._cache[idx]

    def __iter__(self) -> CacheIter[T]:
        # Return self so iteration goes through our __next__ and gets cached.
        return self

    def __next__(self) -> T:
        val = next(self._parent)
        self._cache[self._i] = val
        self._i += 1
        return val

    def __len__(self) -> int:
        return len(self._parent)

    # Optional Python 2-style alias (harmless in Py3; sometimes convenient)
    def next(self) -> T:  # pragma: no cover
        return self.__next__()

    @property
    def cache(self) -> dict[int, T]:
        return self._cache


def drop(tree: dict, keys: list[str]) -> dict:
    return {k: v for k, v in tree.items() if k not in keys}


class _DropKeyDataset(Sequence[dict]):
    """Dataset wrapper that filters observation keys."""

    def __init__(self, dataset: Sequence[dict], drop_keys: Sequence[str] = ()):
        self._dataset = dataset
        self._drop_keys = set(drop_keys)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        traj = self._dataset[index]
        if not self._drop_keys:
            return traj
        return _drop_observation_keys(traj, self._drop_keys)


def _drop_observation_keys(traj: dict, drop_keys: set[str]) -> dict:
    obs = traj.get("observation")
    if not isinstance(obs, dict) or not drop_keys:
        return traj

    filtered: dict[str, Any] = {}
    for key, value in obs.items():
        if _should_drop(drop_keys, key):
            continue
        if isinstance(value, dict):
            if key == "proprio":
                nested = {
                    sub_key: sub_val
                    for sub_key, sub_val in value.items()
                    if not _should_drop(drop_keys, f"{key}_{sub_key}", sub_key, f"{key}/{sub_key}")
                }
                if nested:
                    filtered[key] = nested
                continue
            filtered[key] = value
            continue
        filtered[key] = value

    new_traj = dict(traj)
    new_traj["observation"] = filtered
    return new_traj


def _should_drop(drop_keys: set[str], *candidates: str) -> bool:
    for candidate in candidates:
        if candidate is None:
            continue
        for alias in _expand_aliases(candidate):
            if alias in drop_keys:
                return True
    return False


def _expand_aliases(key: str) -> set[str]:
    if not key:
        return {""}

    aliases = {key}
    cleaned = key.lstrip("/")
    aliases.add(cleaned)

    if "_" in cleaned:
        prefix, suffix = cleaned.split("_", 1)
        aliases.add(suffix)
        if prefix:
            aliases.add(f"{prefix}/{suffix}")

    if "/" in cleaned:
        prefix, suffix = cleaned.split("/", 1)
        if prefix:
            aliases.add(f"{prefix}_{suffix}")
        aliases.add(suffix)

    aliases.add(cleaned.split("/")[-1])

    return {alias for alias in aliases if alias}
