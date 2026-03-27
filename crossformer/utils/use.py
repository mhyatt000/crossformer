"""Custom asdict that respects field metadata for selective serialization.

from module.utils import use

@dataclass
class Config:
    # config settings

    thing: float = 1.0

    use: bool = use.field(default=True)

data = use.asdict(cfg) # doesnt serialize fields with metadata marking
"""

from __future__ import annotations

import copy
from dataclasses import field, fields, is_dataclass
from functools import partial

field = partial(field, metadata={"asdict": False})


def _is_dataclass_instance(obj):
    """Check if obj is a dataclass instance (not the class itself)."""
    return is_dataclass(obj) and not isinstance(obj, type)


_ATOMIC_TYPES = frozenset((int, str, float, complex, bool, bytes, type(None)))


def asdict(obj, *, dict_factory=dict):
    """Return the fields of a dataclass instance as a new dictionary mapping
    field names to field values.

    Example usage::

      @dataclass
      class C:
          x: int
          y: int

      c = C(1, 2)
      assert asdict(c) == {'x': 1, 'y': 2}

    If given, 'dict_factory' will be used instead of built-in dict.
    The function applies recursively to field values that are
    dataclass instances. This will also look into built-in containers:
    tuples, lists, and dicts. Other objects are copied with 'copy.deepcopy()'.
    """
    if not _is_dataclass_instance(obj):
        raise TypeError("asdict() should be called on dataclass instances")
    return _asdict_inner(obj, dict_factory)


def _asdict_dc(obj, dict_factory):
    if dict_factory is dict:
        return {
            f.name: _asdict_inner(getattr(obj, f.name), dict) for f in fields(obj) if f.metadata.get("asdict", True)
        }
    result = [
        (f.name, _asdict_inner(getattr(obj, f.name), dict_factory))
        for f in fields(obj)
        if f.metadata.get("asdict", True)
    ]
    return dict_factory(result)


def _asdict_dict(obj, dict_factory):
    if hasattr(type(obj), "default_factory"):
        result = type(obj)(getattr(obj, "default_factory"))
        for k, v in obj.items():
            result[_asdict_inner(k, dict_factory)] = _asdict_inner(v, dict_factory)
        return result
    return type(obj)((_asdict_inner(k, dict_factory), _asdict_inner(v, dict_factory)) for k, v in obj.items())


def _asdict_inner(obj, dict_factory):
    if type(obj) in _ATOMIC_TYPES:
        return obj
    if _is_dataclass_instance(obj):
        return _asdict_dc(obj, dict_factory)
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):
        return type(obj)(*[_asdict_inner(v, dict_factory) for v in obj])
    if isinstance(obj, (list, tuple)):
        return type(obj)(_asdict_inner(v, dict_factory) for v in obj)
    if isinstance(obj, dict):
        return _asdict_dict(obj, dict_factory)
    return copy.deepcopy(obj)
