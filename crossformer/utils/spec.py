from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
import importlib
from pathlib import Path
from typing import Any, Hashable, Iterable, TypedDict

import jax
from orbax import checkpoint as ocp


@dataclass(frozen=True)
class SimpleSpec:
    shape: tuple[int, ...]
    dtype: Any


SpecValue = tuple[Iterable[int], Any] | SimpleSpec
Spec = dict[Hashable, SpecValue]


def spec(tree: dict[str, Any], simple: bool = True) -> Spec:
    """Create a spec dictionary for the given tree structure."""

    sd = ocp.utils.to_shape_dtype_struct

    def toshape(x: Any) -> Any:
        if not getattr(x, "shape", None):
            return x
        return SimpleSpec(tuple(x.shape), x.dtype)

    return jax.tree.map(sd if not simple else toshape, tree)


def _norm_shape(s: Any) -> tuple[int, ...]:
    # Accept list/tuple/np shape-like → tuple[int,...]
    return tuple(s) if s is not None else ()


def _norm_dtype(dt: Any) -> str:
    # Works for strings, numpy/jax dtypes, and objects with .name or .__name__
    if dt is None:
        return "None"
    if hasattr(dt, "name"):  # numpy/jax dtype
        return str(dt.name)
    if hasattr(dt, "__name__"):  # Python types like int/float
        return dt.__name__
    return str(dt)


def _shape_dtype(value: SpecValue) -> tuple[Any, Any]:
    if isinstance(value, SimpleSpec):
        return value.shape, value.dtype
    return value


def diff(a: Spec, b: Spec, simple: bool = True) -> dict[str, Any]:
    """
    Compare two flat specs {key: (shape, dtype)} and report added/removed/changed.
    Returns:
        {
          "added":   {k: (shape, dtype)},
          "removed": {k: (shape, dtype)},
          "changed": {k: {"from": (shape, dtype), "to": (shape, dtype)}},
        }
    """
    keys_a, keys_b = set(a), set(b)

    added = {k: b[k] for k in (keys_b - keys_a)}
    removed = {k: a[k] for k in (keys_a - keys_b)}

    changed = {}
    for k in keys_a & keys_b:
        sa, da = _shape_dtype(a[k])
        sb, db = _shape_dtype(b[k])
        if _norm_shape(sa) != _norm_shape(sb) or _norm_dtype(da) != _norm_dtype(db):
            changed[k] = {"from": a[k], "to": b[k]}

    return {"added": added, "removed": removed, "changed": changed}


def ezdiff(a: dict[str, Any], b: dict[str, Any], simple: bool = True) -> None:
    from crossformer.utils.tree import flat

    spec_a = spec(flat(a), simple=simple)
    spec_b = spec(flat(b), simple=simple)
    from rich.pretty import pprint

    pprint(diff(spec_a, spec_b, simple=simple))


def valdiff(a: dict[str, Any], b: dict[str, Any], *, atol: float = 1e-5, rtol: float = 1e-5) -> dict[str, Any]:
    """Compare two trees element-wise and report per-key value differences.

    Returns a dict with:
      - ``match``: keys where values are close (within atol/rtol)
      - ``mismatch``: keys where values differ, with max_abs_diff and shapes
      - ``a_only`` / ``b_only``: keys present in only one tree
    """
    import numpy as _np

    from crossformer.utils.tree import flat

    fa, fb = flat(a), flat(b)
    keys_a, keys_b = set(fa), set(fb)

    result: dict[str, Any] = {
        "match": {},
        "mismatch": {},
        "a_only": sorted(keys_a - keys_b),
        "b_only": sorted(keys_b - keys_a),
    }

    for k in sorted(keys_a & keys_b):
        va, vb = _np.asarray(fa[k]), _np.asarray(fb[k])
        if va.shape != vb.shape:
            result["mismatch"][k] = {"reason": "shape", "a": va.shape, "b": vb.shape}
            continue
        if va.dtype.kind in ("U", "S", "O") or vb.dtype.kind in ("U", "S", "O"):
            eq = _np.array_equal(va, vb)
            result["match" if eq else "mismatch"][k] = {"equal": eq}
            continue
        if _np.allclose(va, vb, atol=atol, rtol=rtol, equal_nan=True):
            result["match"][k] = True
        else:
            diff_abs = _np.abs(va.astype(float) - vb.astype(float))
            result["mismatch"][k] = {
                "max_abs_diff": float(diff_abs.max()),
                "mean_abs_diff": float(diff_abs.mean()),
                "a_range": (float(va.min()), float(va.max())),
                "b_range": (float(vb.min()), float(vb.max())),
            }

    return result


def ezvaldiff(a: dict[str, Any], b: dict[str, Any], *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    """Pretty-print a value-level diff between two nested dicts."""
    from rich.pretty import pprint

    result = valdiff(a, b, atol=atol, rtol=rtol)
    n_match = len(result["match"])
    n_mis = len(result["mismatch"])
    print(f"  {n_match} keys match, {n_mis} keys mismatch")
    if result["a_only"]:
        print(f"  a_only: {result['a_only']}")
    if result["b_only"]:
        print(f"  b_only: {result['b_only']}")
    if result["mismatch"]:
        pprint(result["mismatch"])


class ModuleSpec(TypedDict):
    """A JSON-serializable representation of a function or class with some default args and kwargs to pass to
    it. Useful for specifying a particular class or function in a config file, while keeping it serializable
    and overridable from the command line using ml_collections.

    Usage:

        # Preferred way to create a spec:
        >>> from crossformer.model.components.transformer import Transformer
        >>> spec = ModuleSpec.create(Transformer, num_layers=3)
        # Same as above using the fully qualified import string:
        >>> spec = ModuleSpec.create("crossformer.model.components.transformer:Transformer", num_layers=3)

        # Usage:
        >>> ModuleSpec.instantiate(spec) == partial(Transformer, num_layers=3)
        # can pass additional kwargs at instantiation time
        >>> transformer = ModuleSpec.instantiate(spec, num_heads=8)

    Note: ModuleSpec is just an alias for a dictionary (that is strongly typed), not a real class. So from
    your code's perspective, it is just a dictionary.

    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    """

    module: str
    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


def create_module_spec(callable_or_full_name: str | Callable[..., Any], *args: Any, **kwargs: Any) -> ModuleSpec:
    """Create a module spec from a callable or import string."""
    if isinstance(callable_or_full_name, str):
        assert callable_or_full_name.count(":") == 1, (
            "If passing in a string, it must be a fully qualified import string "
            "(e.g. 'crossformer.model.components.transformer:Transformer')"
        )
        module, name = callable_or_full_name.split(":")
    else:
        obj, args, kwargs = _unwrap_partial(callable_or_full_name, args, kwargs)
        module, name = _infer_full_name(obj)

    return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)


def instantiate_module_spec(spec: ModuleSpec) -> partial[Any]:
    if set(spec.keys()) != {"module", "name", "args", "kwargs"}:
        raise ValueError(
            f"Expected ModuleSpec, but got {spec}. ModuleSpec must have keys 'module', 'name', 'args', and 'kwargs'."
        )
    cls = _import_from_string(spec["module"], spec["name"])
    return partial(cls, *spec["args"], **spec["kwargs"])


def module_spec_to_string(spec: ModuleSpec) -> str:
    args = ", ".join(str(arg) for arg in spec["args"])
    kwargs = ", ".join(f"{k}={v}" for k, v in spec["kwargs"].items())
    sep = ", " if args and kwargs else ""
    return f"{spec['module']}:{spec['name']}({args}{sep}{kwargs})"


setattr(ModuleSpec, "create", staticmethod(create_module_spec))
setattr(ModuleSpec, "instantiate", staticmethod(instantiate_module_spec))
setattr(ModuleSpec, "to_string", staticmethod(module_spec_to_string))


@dataclass
class ModuleFile:
    """Load ``ModuleSpec``s from a YAML file in one go.

    A *spec entry* is a mapping that identifies a callable via either:
      - ``module`` + ``name`` keys, or
      - a ``_target_`` string of the form ``"pkg.module:Name"`` (split on ``:``).

    Any remaining keys in that mapping become ``kwargs`` (inline style), while
    explicit ``args`` / ``kwargs`` keys are honored if present.

    ``load`` mirrors the shape of the document:
      - if the top-level node is itself a spec entry, it returns a single
        ``ModuleSpec``;
      - otherwise it traverses the top-level dict/list and returns a matching
        dict/list where any spec-like subentries are replaced by ``ModuleSpec``s
        (non-spec leaves are passed through unchanged).

    Example::

        >>> ModuleFile.load("config/normalize.yaml")
        {'kp3dc_robot': {'module': 'crossformer.data.grain.meta', 'name': 'TMPCLASS',
                         'args': (), 'kwargs': {'agg': 0, 'mask': [...]}}, ...}
    """

    path: Path

    @classmethod
    def load(cls, path: str | Path) -> ModuleSpec | dict[Any, Any] | list[Any]:  # type: ignore[valid-type]
        """Read ``path`` as YAML and build ``ModuleSpec``(s) from it."""
        return cls(Path(path)).parse()

    def read(self) -> Any:
        import yaml

        with open(self.path) as f:
            return yaml.safe_load(f)

    def parse(self) -> ModuleSpec | dict[Any, Any] | list[Any]:  # type: ignore[valid-type]
        return self._parse(self.read())

    @classmethod
    def _parse(cls, node: Any) -> Any:
        spec = cls._as_spec(node)
        if spec is not None:
            return spec
        if isinstance(node, dict):
            return {k: cls._parse(v) for k, v in node.items()}
        if isinstance(node, list):
            return [cls._parse(v) for v in node]
        return node

    @classmethod
    def _as_spec(cls, node: Any) -> ModuleSpec | None:  # type: ignore[valid-type]
        """Return a ``ModuleSpec`` if ``node`` is a spec mapping, else ``None``.

        Nested spec mappings inside ``args``/``kwargs`` are resolved recursively,
        so a spec's arguments may themselves be ``ModuleSpec``s.
        """
        if not isinstance(node, dict):
            return None

        d = dict(node)  # shallow copy — we pop identifying keys off it
        if "_target_" in d:
            target = d.pop("_target_")
            assert isinstance(target, str) and target.count(":") == 1, (
                f"_target_ must be a fully qualified 'module:name' string, got {target!r}"
            )
            module, name = target.split(":")
        elif "module" in d and "name" in d:
            module, name = d.pop("module"), d.pop("name")
        else:
            return None

        args = tuple(cls._parse(a) for a in (d.pop("args", ()) or ()))
        kwargs = {**(d.pop("kwargs", {}) or {}), **d}  # explicit kwargs + inline keys
        kwargs = {k: cls._parse(v) for k, v in kwargs.items()}  # resolve nested specs
        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)


def _infer_full_name(o: object) -> tuple[str, str]:
    module = getattr(o, "__module__", None)
    name = getattr(o, "__name__", None)
    if isinstance(module, str) and isinstance(name, str):
        return module, name
    raise ValueError(
        f"Could not infer identifier for {o}. "
        "Please pass in a fully qualified import string instead "
        "e.g. 'crossformer.model.components.transformer:Transformer'"
    )


def _unwrap_partial(
    o: object, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[object, tuple[Any, ...], dict[str, Any]]:
    while isinstance(o, partial):
        args = (*o.args, *args)
        kwargs = {**(o.keywords or {}), **kwargs}
        o = o.func
    return o, args, kwargs


def _import_from_string(module_string: str, name: str) -> Any:
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as e:
        raise ValueError(f"Could not import {module_string}:{name}") from e
