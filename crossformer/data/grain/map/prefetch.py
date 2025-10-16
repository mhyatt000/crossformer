from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import contextlib
import operator
from typing import Any, Callable, Iterator, Sequence


def noop(x):
    return x


class PrefetchWrapper(Sequence):
    """
    Wrap a random-access sequence and prefetch items with `t` threads.

    - Iteration yields items in order with a rolling prefetch buffer.
    - __getitem__(int) returns a single item (no prefetch).
    - __getitem__(slice | list | tuple) fetches in parallel and preserves order.
    - Use as a context manager to ensure clean shutdown, or call .close().

    Args:
        seq: Underlying sequence supporting __len__ and __getitem__.
        t: Number of threads for prefetch/parallel fetch.
        prefetch: Max number of in-flight items during iteration.
        transform: Optional function applied to each fetched item (e.g., decode).
        start, stop: Optional iteration window on the sequence.
    """

    def __init__(
        self,
        seq: Sequence,
        t: int = 4,
        prefetch: int = 32,
        transform: Callable[[Any], Any] | None = None,
        start: int = 0,
        stop: int | None = None,
    ) -> None:
        if t < 1:
            raise ValueError("t must be >= 1")
        if prefetch < 1:
            raise ValueError("prefetch must be >= 1")

        self._seq = seq
        self._t = t
        self._prefetch = prefetch
        self._xf = transform or noop
        self._start = max(0, start)
        self._stop = len(seq) if stop is None else min(stop, len(seq))
        if self._start > self._stop:
            self._start = self._stop

        self._pool = ProcessPoolExecutor(max_workers=t)
        self._closed = False

    # ---- Context management / cleanup ----
    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def __enter__(self) -> PrefetchWrapper:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        # Best-effort cleanup if user forgets to close
        with contextlib.suppress(Exception):
            self.close()

    # ---- Sequence protocol ----
    def __len__(self) -> int:
        return self._stop - self._start

    def _fetch_one(self, idx: int):
        return self._xf(self._seq[idx])

    def __getitem__(self, key):
        # Single index
        if not isinstance(key, (slice, list, tuple)):
            i = operator.index(key)
            if i < 0:
                i += len(self)
            if not (0 <= i < len(self)):
                raise IndexError("index out of range")
            return self._fetch_one(self._start + i)

        # Slice -> parallel fetch preserving order
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            if step == 0:
                raise ValueError("slice step cannot be zero")
            indices = list(range(start, stop, step))
        else:  # list/tuple of indices
            indices = [operator.index(i) for i in key]
            # Normalize negatives relative to wrapped view
            indices = [i + len(self) if i < 0 else i for i in indices]
            for i in indices:
                if not (0 <= i < len(self)):
                    raise IndexError("index out of range")

        abs_indices = [self._start + i for i in indices]
        # Small batches: map is fine; large batches still benefit from threads
        futures = [self._pool.submit(self._fetch_one, i) for i in abs_indices]
        return [f.result() for f in futures]

    # ---- Iteration with rolling prefetch ----
    def __iter__(self) -> Iterator[Any]:
        n = len(self)
        if n == 0:
            return
        base = self._start

        # Queue of (abs_index, future), always in order
        q: deque = deque()
        next_submit = 0
        next_yield = 0

        # Prime the buffer
        initial = min(self._prefetch, n)
        for k in range(initial):
            idx = base + k
            q.append((idx, self._pool.submit(self._fetch_one, idx)))
        next_submit = initial

        while next_yield < n:
            idx, fut = q[0]
            # Block only for the next in-order item
            item = fut.result()
            yield item
            q.popleft()
            next_yield += 1

            # Refill to keep up to _prefetch in flight
            while len(q) < self._prefetch and next_submit < n:
                idx2 = base + next_submit
                q.append((idx2, self._pool.submit(self._fetch_one, idx2)))
                next_submit += 1
