from __future__ import annotations

from typing import Generic, Hashable, Sequence, TypeVar

from grain._src.python.dataset import dataset
from grain._src.python.dataset.transformations import source

T = TypeVar("T")


class PackBySizeMapSource(source.SourceMapDataset, Generic[T]):
    """Maps episode index -> contiguous list of steps from parent.

    index i -> episode_ids[i]
    returns [parent[start : start+len)]
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        source: dataset.MapDataset[T],
        size_dict: dict[Hashable, int],
        pack_fn: None = None,
        *,
        preserve_insertion_order: bool = True,
    ):
        super().__init__(source)

        self._pack_fn = pack_fn if pack_fn else lambda x: x

        # Episode order
        if preserve_insertion_order:
            self._eids: list[Hashable] = list(size_dict.keys())
        else:
            # Try numeric sort, else lexicographic
            try:
                self._eids = sorted(size_dict.keys(), key=lambda x: int(x))
            except Exception:
                self._eids = sorted(size_dict.keys())

        # Lengths and starts (prefix sums)
        self._lens: list[int] = [int(size_dict[e]) for e in self._eids]
        self._starts: list[int] = []
        acc = 0
        for L in self._lens:
            self._starts.append(acc)
            acc += L
        self._total_len = acc  # total steps across all episodes

        # Quick tuples for indexing
        self._triples: list[tuple[int, int, Hashable]] = [
            (self._starts[i], self._lens[i], self._eids[i]) for i in range(len(self._eids))
        ]

    def __len__(self) -> int:
        return len(self._eids)

    def __str__(self) -> str:
        return "PackBySizeMapDataset"

    def episode_ids(self) -> Sequence[Hashable]:
        return tuple(self._eids)

    def episode_start(self, eid: Hashable) -> int:
        idx = self._eids.index(eid)
        return self._starts[idx]

    def episode_len(self, eid: Hashable) -> int:
        idx = self._eids.index(eid)
        return self._lens[idx]

    def __getitem__(self, index: int) -> list[T]:
        if isinstance(index, slice):
            return self.slice(index)
        if index < 0 or index >= len(self._eids):
            raise IndexError(index)

        with self._stats.record_self_time():
            start, L, _eid = self._triples[index]
            # Pull contiguous steps
            out = [start + j for j in range(L)]
            out = self._source.__getitems__(out)
            return self._stats.record_output_spec(self._pack_fn(out))


class PackBySizeMapDataset(dataset.MapDataset, Generic[T]):
    """Maps episode index -> contiguous list of steps from parent.

    index i -> episode_ids[i]
    returns [parent[start : start+len)]
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        parent: dataset.MapDataset[T],
        size_dict: dict[Hashable, int],
        pack_fn: None = None,
        *,
        preserve_insertion_order: bool = True,
    ):
        super().__init__(parent)

        self._pack_fn = pack_fn if pack_fn else lambda x: x
        size_dict = {int(k): v for k, v in size_dict.items()}
        self.eid2idx = {eid: idx for idx, eid in enumerate(size_dict.keys())}

        # Episode order
        if preserve_insertion_order:
            self._eids: list[Hashable] = list(size_dict.keys())
        else:
            # Try numeric sort, else lexicographic
            try:
                self._eids = sorted(size_dict.keys(), key=lambda x: int(x))
            except Exception:
                self._eids = sorted(size_dict.keys())

        # Lengths and starts (prefix sums)
        self._lens: list[int] = [int(size_dict[e]) for e in self._eids]
        self._starts: list[int] = []
        acc = 0
        for L in self._lens:
            self._starts.append(acc)
            acc += L
        self._total_len = acc  # total steps across all episodes

        # Quick tuples for indexing
        self._triples: list[tuple[int, int, Hashable]] = [
            (self._starts[i], self._lens[i], self._eids[i]) for i in range(len(self._eids))
        ]

    def __len__(self) -> int:
        return len(self._eids)

    def __str__(self) -> str:
        return "PackBySizeMapDataset"

    def episode_ids(self) -> Sequence[Hashable]:
        return tuple(self._eids)

    def episode_start(self, eid: Hashable) -> int:
        idx = self._eids.index(eid)
        return self._starts[idx]

    def episode_len(self, eid: Hashable) -> int:
        idx = self._eids.index(eid)
        return self._lens[idx]

    def __getitem__(self, index: int) -> list[T]:
        if isinstance(index, slice):
            return self.slice(index)
        if index < 0 or index >= len(self._eids):
            raise IndexError(index)
        with self._stats.record_self_time():
            start, L, _eid = self._triples[index]
            # Pull contiguous steps
            out = [start + j for j in range(L)]
            out = [self._parent[i] for i in out]
            return self._stats.record_output_spec(self._pack_fn(out))
