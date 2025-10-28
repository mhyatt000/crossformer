from __future__ import annotations

from collections import defaultdict, OrderedDict
import threading
from typing import Callable, Generic, Hashable, TypeVar

from grain._src.python.dataset import dataset
from readerwriterlock import rwlock
from rich.pretty import pprint

T = TypeVar("T")
U = TypeVar("U")


class CacheByKeyMapDataset(dataset.MapDataset[U], Generic[T, U]):
    """Caches parent[index] (optionally transformed) under a user key.

    If the same key is requested again, returns the cached value without
    re-reading the parent. Cache is per-process.

    Args:
      parent: MapDataset producing elements of type T.
      key_fn: index -> key. Default uses the index itself.
      value_fn: (index, element) -> value to cache. Default caches `element`.
      max_items: LRU capacity. None means unbounded.
      enabled: toggle caching.
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        parent: dataset.MapDataset[T],
        *,
        key_fn: Callable[[int], Hashable] | None = None,
        value_fn: Callable[[int, T], U] | None = None,
        max_items: int | None = None,
        enabled: bool = True,
    ):
        super().__init__(parent)
        self._key_fn = key_fn or (lambda i: i)
        self._value_fn = value_fn or (lambda i, x: x)  # type: ignore[return-value]
        self._enabled = bool(enabled)
        self._max = max_items if (max_items is None or max_items > 0) else None
        self._cache: OrderedDict[Hashable, U] = OrderedDict()
        self._key_lock = defaultdict(threading.Lock)  # one lock per key
        self._meta_lock = threading.RLock()
        self._rwl = rwlock.RWLockWrite()

    def __len__(self) -> int:
        return len(self._parent)

    def __str__(self) -> str:
        return "CacheByKeyMapDataset"

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)

    def _cache_get(self, key: Hashable) -> U | None:
        if not self._enabled:
            return None
        val = self._cache.get(key)
        return val

    def _cache_put(self, key: Hashable, value: U) -> U:
        if not self._enabled:
            return value
        self._cache[key] = value
        if self._max is not None and len(self._cache) > self._max:  # Evict oldest
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> U:
        if isinstance(index, slice):
            return self.slice(index)

        index = index % len(self)
        with self._stats.record_self_time():
            with self._rwl.gen_rlock():
                key = self._key_fn(index)
                hit = self._cache_get(key)
            if hit is not None:
                return self._stats.record_output_spec(hit)
            pprint((len(self._cache), "/", len(self)))

            elem = self._parent[index]
            val: U = self._value_fn(index, elem)
            lk = self._key_lock[key]
            with lk:
                val = self._cache_put(key, val)
            return self._stats.record_output_spec(val)
