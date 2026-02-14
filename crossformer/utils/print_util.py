from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import field, fields, is_dataclass
from typing import Any, Callable, overload, TypeVar

import numpy as np
from rich import print


def _fmt_np(a: np.ndarray) -> str:
    return f"ndarray(shape={a.shape}, dtype={a.dtype})"


def _brief(v: Any, *, depth: int, max_depth: int) -> str:
    if depth >= max_depth:
        return "..."

    if isinstance(v, np.ndarray):
        return _fmt_np(v)
        # return _np_pretty(v)

    if is_dataclass(v) and not isinstance(v, type):
        # nested dataclass: use its repr (which might also be brief)
        return repr(v)

    if isinstance(v, dict):
        items = ", ".join(
            f"{_brief(k, depth=depth + 1, max_depth=max_depth)}: {_brief(val, depth=depth + 1, max_depth=max_depth)}"
            for k, val in v.items()
        )
        return "{" + items + "}"

    if isinstance(v, (list, tuple)):
        inner = ", ".join(_brief(x, depth=depth + 1, max_depth=max_depth) for x in v)
        if isinstance(v, list):
            return "[" + inner + "]"
        return "(" + inner + ("," if len(v) == 1 else "") + ")"

    if isinstance(v, set):
        inner = ", ".join(_brief(x, depth=depth + 1, max_depth=max_depth) for x in v)
        return "{" + inner + "}"

    return repr(v)


T = TypeVar("T")


@overload
def brief(cls: type[T]) -> type[T]: ...
@overload
def brief(*, max_depth: int = 4, **dc_kwargs) -> Callable[[type[T]], type[T]]: ...


def brief(cls: type[T] | None = None, *, max_depth: int = 4, **dc_kwargs):
    """
    Like @dataclass, but injects a repr that summarizes np.ndarrays as shape/dtype.
    Works even if arrays are assigned after __init__.
    """

    def wrap(c: type[T]) -> type[T]:
        c = _dataclass(c, **dc_kwargs)

        def __repr__(self) -> str:
            parts = []
            for f in fields(self):
                v = getattr(self, f.name)
                parts.append(f"{f.name}={_brief(v, depth=0, max_depth=max_depth)}")
            return f"{self.__class__.__name__}(" + ", ".join(parts) + ")"

        c.__repr__ = __repr__
        # setattr(c, "__repr__", __repr__)

        # c.__str__ = __repr__  # optional; remove if you want str() different
        return c

    return wrap(cls) if cls is not None else wrap


def main():
    # @dataclass
    @brief
    class A:
        a: np.ndarray = field(default_factory=lambda: np.array([1, 2, 3]))
        b: np.ndarray = field(default_factory=lambda: np.array([4, 5, 6]))

    a = A()

    print(a)


if __name__ == "__main__":
    main()
