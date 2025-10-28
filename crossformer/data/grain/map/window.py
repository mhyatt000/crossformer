from __future__ import annotations

from collections import deque
from typing import Callable, Generic, Sequence, TypeVar

from grain._src.python.dataset import dataset, stats

from crossformer.data.grain.transforms import batch_fn

T = TypeVar("T")
U = TypeVar("U")

# -------- MapDataset version (random access with sliding cache) --------


class WindowFnMapDataset(dataset.MapDataset[U], Generic[T, U]):
    """Apply window_fn to a sliding window starting at index i.

    Window is [i, i+window_size). Cache slides by one on sequential access
    to avoid re-fetching elements already in the window.
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        parent: dataset.MapDataset[T],
        *,
        window_size: int,
        window_fn: Callable[[Sequence[T]], U],
    ):
        super().__init__(parent)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self._window_size = int(window_size)
        self._window_fn = window_fn
        # Sliding cache
        self._cache_start: int | None = None
        self._cache: deque[T] = deque()  # maxlen not set due to tail growth logic

    def __len__(self) -> int:
        return len(self._parent)

    def __str__(self) -> str:
        return "WindowFnMapDataset"

    def _refill_cache_from(self, start: int):
        self._cache.clear()
        n = len(self._parent)
        end = min(start + self._window_size, n)
        for j in range(start, end):
            self._cache.append(self._parent[j])
        self._cache_start = start

    def _advance_cache_by_one(self):
        """Slide cache by one: drop left, append next element if exists."""
        assert self._cache_start is not None
        n = len(self._parent)
        next_idx = self._cache_start + len(self._cache)  # index to append
        if self._cache:
            self._cache.popleft()
        if next_idx < n:
            self._cache.append(self._parent[next_idx])
        self._cache_start += 1

    def __getitem__(self, index: int) -> U:
        if isinstance(index, slice):
            return self.slice(index)

        with self._stats.record_self_time():
            n = len(self._parent)
            if index < 0 or index >= n:
                raise IndexError(index)

            # Ensure cache covers [index, index+window_size)
            if self._cache_start is None:
                self._refill_cache_from(index)
            elif index == self._cache_start + 1:
                # Sequential access -> cheap slide
                self._advance_cache_by_one()
            elif not (self._cache_start <= index < self._cache_start + len(self._cache)):
                # Jump -> rebuild
                self._refill_cache_from(index)
            else:
                # Within current window but not the first element:
                # Realign so that index becomes the new window start by sliding k steps.
                k = index - self._cache_start
                for _ in range(k):
                    self._advance_cache_by_one()

            # Apply fn to current window view (may be shorter at dataset tail)
            out = self._window_fn(list(self._cache))
            return self._stats.record_output_spec(out)


# -------- IterDataset version (sequential with sliding window) --------


class WindowFnIterDataset(dataset.IterDataset[U], Generic[T, U]):
    """Apply window_fn to a sliding window over an iterator.

    On each next(): emit window_fn(window) for the current window,
    then slide by one (pop left, push next from parent).
    """

    def __init__(
        self,
        parent: dataset.IterDataset[T],
        *,
        window_size: int,
        window_fn: Callable[[Sequence[T]], U],
    ):
        super().__init__(parent)
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        self._window_size = int(window_size)
        self._window_fn = window_fn

    def __iter__(self) -> _WindowFnDatasetIterator[T, U]:
        parent_iter = self._parent.__iter__()
        return _WindowFnDatasetIterator(parent_iter, window_size=self._window_size, window_fn=self._window_fn)

    def __str__(self) -> str:
        return "WindowFnIterDataset"


class _WindowFnDatasetIterator(dataset.DatasetIterator[U], Generic[T, U]):
    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        parent: dataset.DatasetIterator[T],
        *,
        window_size: int,
        window_fn: Callable[[Sequence[T]], U],
    ):
        super().__init__(parent)
        self._window_size = window_size
        self._window_fn = window_fn
        self._window: deque[T] = deque()
        self._parent_window_start_iter_state = self._parent.get_state()
        self._pos_from_start = 0
        self._init = True
        self._parent_exhausted = False

    def _maybe_mark_start(self):
        if self._init:
            self._init = False
        else:
            self._pos_from_start += 1

    def _fill_until_full_or_eof(self):
        try:
            while len(self._window) < self._window_size:
                self._window.append(next(self._parent))
        except StopIteration:
            self._parent_exhausted = True

    def _prime_for_next(self):
        # If empty, capture state and fill a fresh window.
        if not self._window:
            if self._parent_exhausted:
                return False
            self._parent_window_start_iter_state = self._parent.get_state()
            self._maybe_mark_start()
            self._fill_until_full_or_eof()
        return bool(self._window)

    @stats.record_next_duration_if_output
    def __next__(self) -> U:
        if not self._prime_for_next():
            raise StopIteration

        # Emit on current window snapshot.
        out = self._window_fn(list(self._window))

        # Slide by one.
        if self._window:
            self._window.popleft()
        if not self._parent_exhausted:
            try:
                self._window.append(next(self._parent))
            except StopIteration:
                self._parent_exhausted = True
        return out

    def get_state(self):
        return {
            "parent_window_start_state": self._parent_window_start_iter_state,
            "pos_from_start": self._pos_from_start,
            "parent_exhausted": self._parent_exhausted,
        }

    def set_state(self, state):
        self._parent_window_start_iter_state = state["parent_window_start_state"]
        self._parent.set_state(self._parent_window_start_iter_state)
        self._pos_from_start = state["pos_from_start"]
        self._parent_exhausted = state["parent_exhausted"]

        # Rebuild window deterministically from the stored start.
        self._window.clear()
        self._fill_until_full_or_eof()
        for _ in range(min(self._pos_from_start, len(self._window))):
            self._window.popleft()
            if not self._parent_exhausted:
                try:
                    self._window.append(next(self._parent))
                except StopIteration:
                    self._parent_exhausted = True

    def __str__(self) -> str:
        return "WindowFnDatasetIterator"


def mk_chunk(a: int, o: int = 1):
    assert o == 1, "only support o=1 for now"

    def chunk_fn(window):
        # Assume pytree with keys "action" and "observation"
        # Batch first a/o entries across the window
        first = window[0]
        actions = batch_fn([w["action"] for w in window[:a]])
        # action_pad_mask = 1 if episode_id is the same as first else 0
        get_id = lambda w: w["info"]["id"]["episode_id"]
        action_pad_mask = batch_fn([get_id(w) == get_id(first) for w in window[:a]])
        # augmax needs same window for obs and task
        # obs     = batch_fn([w["observation"] for w in window[:o]])
        return {
            **first,
            "action": actions,
            "action_pad_mask": action_pad_mask,
            # "observation": obs,
        }

    return chunk_fn
