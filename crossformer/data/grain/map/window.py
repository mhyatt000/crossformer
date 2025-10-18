from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Callable, Generic, Iterable, Iterator, Mapping, Sequence, TypeVar

import grain
from grain.experimental import FlatMapTransform
import jax
from jax import tree_util

from crossformer.data.grain.util.deco import timeit

T = TypeVar("T")
U = TypeVar("U")


# If your Grain import path differs, adjust the base class import below:


def _first_dim_len(x: Any) -> int | None:
    """Return length of the first (time) dimension if array-like, else None."""
    # numpy / jax / torch / tf all expose .shape; python lists expose __len__
    if hasattr(x, "shape") and isinstance(getattr(x, "shape"), (tuple, list)):
        return int(x.shape[0]) if len(x.shape) > 0 else None
    if isinstance(x, (list, tuple)):
        return len(x) if len(x) > 0 else 0
    return None


def _collect_first_dim_lens(tree: Any, acc: list[int]) -> None:
    """Collect candidate time lengths from all leaves in a nested py-tree."""
    if isinstance(tree, Mapping):
        for v in tree.values():
            _collect_first_dim_lens(v, acc)
    elif isinstance(tree, (list, tuple)):
        for v in tree:
            _collect_first_dim_lens(v, acc)
    else:
        n = _first_dim_len(tree)
        if n is not None:
            acc.append(n)


def _mode_length(lengths: list[int]) -> int:
    """Return the most common (>0) first-dimension length as T."""
    lengths = [n for n in lengths if n and n > 0]
    if not lengths:
        raise ValueError("Could not infer time length T from episode leaves.")
    c = Counter(lengths)
    return c.most_common(1)[0][0]


def _slice_window_with_tree(episode: Any, start: int, stop: int, T: int) -> Any:
    def slicer(leaf):
        # leaf might be numpy / jax / torch / tf / list-like, etc.
        # try to detect first dimension length
        # Note: JAX arrays / numpy have .shape
        if hasattr(leaf, "shape"):
            if len(leaf.shape) > 0 and leaf.shape[0] == T:
                return leaf[start:stop]
            else:
                return leaf
        # fallback to Python sequences
        try:
            # only slice if len matches T
            if hasattr(leaf, "__len__") and len(leaf) == T:
                return leaf[start:stop]
        except Exception:
            pass
        return leaf

    return tree_util.tree_map(slicer, episode)


@dataclass
class WindowedFlatMap(FlatMapTransform):
    """
    Grain FlatMap that emits fixed-length sliding windows of an episode.

    - `size`: required window length.
    - `stride`: hop length between starts (default 1).
    - Only full windows are emitted (drop remainder).
    """

    size: int
    stride: int = 1

    def len(self, L):
        return 0 if self.size > L else (L - self.size) // self.stride + 1

    def flat_map(self, episode: dict):
        # 1) Infer T (time length) robustly from the episode py-tree.
        lengths: list[int] = []
        _collect_first_dim_lens(episode, lengths)
        T = _mode_length(lengths)

        if self.size <= 0:
            raise ValueError(f"`size` must be > 0, got {self.size}")
        if self.stride <= 0:
            raise ValueError(f"`stride` must be > 0, got {self.stride}")

        # 2) Yield sliding windows [t, t+size) where t+size <= T.
        stop_max = T - self.size + 1
        for start in range(0, max(0, stop_max), self.stride):
            stop = start + self.size
            yield _slice_window_with_tree(episode, start, stop, T)
            # yield _slice_window(episode, start, stop, T)


class _MyFlatMapTransform(Generic[T, U]):
    """
    Interface: must implement flat_map(elem: T) -> Iterable[U]
    """

    def flat_map(self, elem: T) -> Iterable[U]:
        raise NotImplementedError


class FlatMapDataset(Generic[T, U], Iterable[U]):
    """
    A dataset that wraps a parent dataset and applies a FlatMapTransform
    to each element, flattening the resulting streams.
    """

    def __init__(self, parent: Iterable[T], tf: FlatMapTransform, L: int, seek: Callable | None = None):
        # We can reuse parent's transforms / configuration
        # super().__init__(parent)
        # or if MapDataset has a different constructor, adapt accordingly
        self.parent = parent
        self.tf = tf
        self.L = L
        self.seek = seek

    def __len__(self) -> int:
        return self.L

    def __getitem__(self, index):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[U]:
        for elem in self.parent:
            outs = self.tf.flat_map(elem)
            # allow flat_map to yield any iterable (list, generator, etc)
            yield from outs


@dataclass
class MyFlatMap(grain.experimental.FlatMapTransform):
    max_fan_out: int = 1000

    def flat_map(self, element: dict) -> list[dict]:
        n = len(element["info"]["step_id"])
        return [jax.tree.map(lambda x: x[i], element) for i in range(n)]


class FlattenTreeDataset(Generic[T, U], Iterable[U]):
    def __init__(self, parent: Iterable[T], lengths: Sequence[int]):
        self.parent = parent
        self.L = sum(lengths)
        self.lengths = lengths

        self.idx_map = {}
        self.build_map()
        self.pcache = {}

    def tf(self, tree: dict, j: int):
        return jax.tree.map(lambda x: x[j], tree)

    def __len__(self) -> int:
        return self.L

    def build_map(self):
        """seeks the correct int after flattening"""
        if self.idx_map is None:
            return

        # make dict where k 0->L-1, v is (i,j) where i is episode index, j is index in episode
        self.idx_map, count = {}, 0
        for i, eplen in enumerate(self.lengths):
            for j in range(eplen):
                self.idx_map[count] = (i, j)
                count += 1

    def seek(self, index) -> tuple[int, int]:
        return self.idx_map[index]

    def manage_cache(self, max_size: int = 100):
        """simple cache management to avoid memory issues"""
        if len(self.pcache) > max_size:
            # pop a random key
            self.pcache.pop(next(iter(self.pcache)))

    @timeit
    def __getitem__(self, index):
        # overrides __iter__ and its slow because it cannot prefetch episodes (i think)
        i, j = self.seek(index)

        episode = self.pcache.get(i, self.parent[i])
        self.pcache[i] = episode
        self.manage_cache()

        return self.tf(episode, j)

    def __iter__(self) -> Iterator[U]:
        return self

    def __next__(self) -> U:
        for i, j in self.idx_map.values():
            episode = self.parent[i]
            yield self.tf(episode, j)
        raise StopIteration


class WindowFlatDataset(FlatMapDataset):
    def __init__(self, parent: Iterable[T], w: WindowedFlatMap):
        assert getattr(parent, "lengths", None) is not None, (
            "Parent dataset must have lengths() method to estimate total length."
        )
        self._episode_lengths = list(parent.lengths())
        self._win_counts = [w.len(_l) for _l in self._episode_lengths]
        self._cumulative = list(accumulate(self._win_counts))
        total = sum(self._win_counts)
        super().__init__(parent, w, total)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("WindowFlatDataset indices must be integers")

        length = self.L
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(index)

        episode_idx = bisect_right(self._cumulative, index)
        prev = self._cumulative[episode_idx - 1] if episode_idx > 0 else 0
        offset = index - prev

        stride = self.tf.stride
        size = self.tf.size
        start = offset * stride
        stop = start + size

        episode = self.parent[episode_idx]
        T = self._episode_lengths[episode_idx]
        if stop > T:
            raise IndexError("Window exceeds episode length")
        return _slice_window_with_tree(episode, start, stop, T)
