from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import types

import jax
from jax._src import config as jax_internal_config
import jax.numpy as jnp
import numpy as np

# Flax expects newer JAX builds that expose define_bool_state; alias to the
# modern equivalent when running tests on newer releases.
if not hasattr(jax.config, "define_bool_state"):
    fallback = getattr(jax_internal_config, "bool_state", None)
    if fallback is None:
        fallback = jax_internal_config.define_bool_state
    jax.config.define_bool_state = fallback

# Recent JAX versions removed jax.experimental.maps, but Flax still imports it.
if "jax.experimental.maps" not in sys.modules:
    maps_module = types.ModuleType("jax.experimental.maps")
    empty_mesh = types.SimpleNamespace(devices=types.SimpleNamespace(shape=()))
    maps_module.thread_resources = types.SimpleNamespace(env=types.SimpleNamespace(physical_mesh=empty_mesh))
    sys.modules["jax.experimental.maps"] = maps_module
    if hasattr(jax, "experimental"):
        setattr(jax.experimental, "maps", maps_module)

if not hasattr(jax.nn, "normalize"):

    def _normalize(x, axis=-1, epsilon=1e-12):
        norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + epsilon)

    jax.nn.normalize = _normalize

if "tensorflow" not in sys.modules:
    tf_module = types.ModuleType("tensorflow")

    class _Tensor(np.ndarray):
        def __new__(cls, input_array, tf_dtype=None):
            obj = np.asarray(input_array).view(cls)
            obj._tf_dtype = tf_dtype
            return obj

        @property
        def dtype(self):
            return getattr(self, "_tf_dtype", None) or super().dtype

        def numpy(self):
            return np.asarray(self)

    def _to_tensor(value, dtype=None):
        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        array = np.array(value, dtype=np_dtype) if dtype is not None else np.array(value)
        return _Tensor(array, tf_dtype=dtype)

    def _uniform(shape, minval=0.0, maxval=1.0, dtype=np.float32):
        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        data = np.random.uniform(minval, maxval, size=shape).astype(np_dtype)
        return _Tensor(data, tf_dtype=dtype)

    def _range(start, limit=None, delta=1, dtype=np.int32):
        if limit is None:
            start, limit = 0, start
        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        return _Tensor(np.arange(start, limit, delta, dtype=np_dtype), tf_dtype=dtype)

    def _asarray(value):
        return np.asarray(value)

    def _tensor_from(value, dtype=None):
        if dtype is None:
            return _Tensor(np.asarray(value))
        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        return _Tensor(np.asarray(value, dtype=np_dtype), tf_dtype=dtype)

    tf_module.float32 = np.float32
    tf_module.bool = np.bool_
    tf_module.int32 = np.int32
    tf_module.Tensor = _Tensor
    tf_module.Variable = _Tensor
    tf_module.string = np.bytes_

    tf_module.random = types.SimpleNamespace(
        uniform=_uniform,
        shuffle=lambda x: _Tensor(
            np.random.permutation(_asarray(x)),
            tf_dtype=getattr(x, "dtype", None),
        ),
    )
    tf_module.ones = lambda shape, dtype=np.float32: _tensor_from(
        np.ones(shape, dtype=getattr(dtype, "as_numpy_dtype", dtype)), dtype
    )
    tf_module.ones_like = lambda value, dtype=None: _tensor_from(
        np.ones_like(_asarray(value), dtype=getattr(dtype, "as_numpy_dtype", dtype)),
        dtype,
    )
    tf_module.zeros_like = lambda value, dtype=None: _tensor_from(
        np.zeros_like(_asarray(value), dtype=getattr(dtype, "as_numpy_dtype", dtype)),
        dtype,
    )
    tf_module.zeros = lambda shape, dtype=np.float32: _tensor_from(
        np.zeros(shape, dtype=getattr(dtype, "as_numpy_dtype", dtype)), dtype
    )
    tf_module.constant = lambda value, dtype=None: _to_tensor(value, dtype)
    tf_module.shape = lambda value: np.shape(value)
    tf_module.range = _range

    def _dtype_of(*values, default=None):
        for value in values:
            if isinstance(value, _Tensor) and getattr(value, "_tf_dtype", None) is not None:
                return value._tf_dtype
        return default

    tf_module.maximum = lambda x, y: _tensor_from(np.maximum(_asarray(x), _asarray(y)), _dtype_of(x, y))
    tf_module.minimum = lambda x, y: _tensor_from(np.minimum(_asarray(x), _asarray(y)), _dtype_of(x, y))
    tf_module.where = lambda cond, x, y: _tensor_from(
        np.where(_asarray(cond), _asarray(x), _asarray(y)), _dtype_of(x, y)
    )
    tf_module.logical_and = lambda x, y: _tensor_from(np.logical_and(_asarray(x), _asarray(y)), np.bool_)
    tf_module.logical_not = lambda x: _tensor_from(np.logical_not(_asarray(x)), np.bool_)
    tf_module.fill = lambda dims, value: _tensor_from(
        np.full(tuple(np.asarray(dims, dtype=int).tolist()), value), type(value)
    )
    tf_module.meshgrid = lambda *args, **kwargs: [_Tensor(arr) for arr in np.meshgrid(*map(_asarray, args), **kwargs)]
    tf_module.concat = lambda values, axis=0: _tensor_from(
        np.concatenate([_asarray(v) for v in values], axis=axis),
        _dtype_of(*values),
    )
    tf_module.pad = lambda tensor, paddings, mode="constant", constant_values=0: _tensor_from(
        np.pad(
            _asarray(tensor),
            [tuple(p) for p in paddings],
            mode=mode,
            constant_values=constant_values,
        ),
        getattr(tensor, "dtype", None),
    )
    tf_module.gather = lambda params, indices, axis=0: _tensor_from(
        np.take(_asarray(params), _asarray(indices), axis=axis), getattr(params, "dtype", None)
    )
    tf_module.map_fn = lambda fn, elems, fn_output_signature=None: _tensor_from(
        np.array([fn(elem) for elem in _asarray(elems)]), fn_output_signature
    )
    tf_module.reduce_any = lambda value, axis=None: np.any(value, axis=axis)
    tf_module.equal = lambda x, y: _tensor_from(np.equal(_asarray(x), _asarray(y)), np.bool_)
    tf_module.reshape = lambda tensor, shape: _tensor_from(
        np.reshape(_asarray(tensor), tuple(shape)), getattr(tensor, "dtype", None)
    )
    tf_module.cast = lambda value, dtype: _tensor_from(
        _asarray(value).astype(getattr(dtype, "as_numpy_dtype", dtype)), dtype
    )

    def _string_length(value):
        arr = _asarray(value)

        def _len(elem):
            if isinstance(elem, (bytes, np.bytes_)):
                return len(elem.decode("utf-8"))
            return len(str(elem))

        vectorized = np.vectorize(_len, otypes=[np.int32])
        return _tensor_from(vectorized(arr), np.int32)

    tf_module.strings = types.SimpleNamespace(length=_string_length)
    tf_module.nest = types.SimpleNamespace(
        map_structure=lambda func, *structures: jax.tree_util.tree_map(lambda *xs: func(*xs), *structures)
    )

    sys.modules["tensorflow"] = tf_module
