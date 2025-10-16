from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, Generic, Iterable, Iterator, Mapping, TypeVar

from grain.experimental import FlatMapTransform
from jax import tree_util

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

    def __init__(self, parent: Iterable[T], tf: FlatMapTransform, L: int):
        # We can reuse parent's transforms / configuration
        # super().__init__(parent)
        # or if MapDataset has a different constructor, adapt accordingly
        self.parent = parent
        self.tf = tf
        self.L = L

    def __len__(self) -> int:
        return self.L

    def __iter__(self) -> Iterator[U]:
        for elem in self.parent:
            outs = self.tf.flat_map(elem)
            # allow flat_map to yield any iterable (list, generator, etc)
            yield from outs


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
