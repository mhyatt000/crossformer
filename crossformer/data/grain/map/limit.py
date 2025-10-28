from __future__ import annotations

from typing import Generic, TypeVar

from grain._src.python.dataset import dataset

T = TypeVar("T")


class LimitDataset(dataset.MapDataset, Generic[T]):
    """Maps episode index -> contiguous list of steps from parent.

    index i -> episode_ids[i]
    returns [parent[start : start+len)]
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(
        self,
        parent: dataset.MapDataset[T],
        limit: int,
    ):
        super().__init__(parent)
        self._limit = limit

    def __len__(self) -> int:
        return min(len(self._parent), self._limit)

    def __str__(self) -> str:
        return "LimitDataset"

    def __getitem__(self, index: int) -> list[T]:
        if isinstance(index, slice):
            return self.slice(index)

        # if idx is greater, do divmod
        index = index % self._limit
        return self._parent[index]
