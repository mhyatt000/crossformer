from __future__ import annotations

from contextlib import contextmanager
import importlib
import logging
import multiprocessing
import os
from types import ModuleType
from typing import Any

_GROUP_SIZE_WARNING = "Grain requires group size 1 for good performance"


def _set_worker_jax_cpu_env() -> None:
    if multiprocessing.parent_process() is None:
        return
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


def import_jax_cpu_safe() -> ModuleType:
    """Import JAX after forcing CPU-only env in spawned workers."""
    _set_worker_jax_cpu_env()
    return importlib.import_module("jax")


@contextmanager
def quiet_jax_xla_bridge():
    """Silence expected `Jax plugin configuration error` and `CUDA_ERROR_NO_DEVICE` noise."""
    logger = logging.getLogger("jax._src.xla_bridge")
    old = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = old


class _SuppressArrayRecordGroupSize(logging.Filter):
    """Drop repeated `Grain requires group size 1 for good performance` warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        return _GROUP_SIZE_WARNING not in record.getMessage()


def install_absl_filter() -> None:
    """Install the `Grain requires group size 1 for good performance` filter once."""
    logger = logging.getLogger("absl")
    if any(isinstance(f, _SuppressArrayRecordGroupSize) for f in logger.filters):
        return
    logger.addFilter(_SuppressArrayRecordGroupSize())


@contextmanager
def silence_absl_errors():
    """Mute expected absl arrayrecord noise while probing dataset-backed sources."""
    logger = logging.getLogger("absl")
    disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = disabled


class SilentDataSource:
    """Wrap a source so len/getitem calls do not spam expected absl arrayrecord warnings."""

    def __init__(self, src: Any) -> None:
        self._src = src
        install_absl_filter()

    def __len__(self) -> int:
        with silence_absl_errors():
            return len(self._src)

    def __getitem__(self, idx: int) -> Any:
        with silence_absl_errors():
            return self._src[idx]

    def __getitems__(self, indices) -> Any:
        with silence_absl_errors():
            return self._src.__getitems__(indices)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._src, name)

    def __getstate__(self) -> dict[str, Any]:
        return {"src": self._src}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._src = state["src"]
        install_absl_filter()

    def __repr__(self) -> str:
        return repr(self._src)


def wrap_grain_source(src: Any) -> Any:
    """Recursively wrap Grain image/proprio sources with quiet datasource shells."""
    if isinstance(src, SilentDataSource):
        return src
    if hasattr(src, "_img") and hasattr(src, "_pro"):
        src._img = wrap_grain_source(src._img)
        src._pro = wrap_grain_source(src._pro)
        return src
    return SilentDataSource(src)


def patch_arec_source() -> None:
    """Patch `Arec.source` so debug probes avoid expected absl arrayrecord warning spam."""
    import crossformer.cn.dataset.mix as mix_mod

    prop = mix_mod.Arec.source
    if getattr(prop.fget, "__name__", "") == "_silent_arec_source":
        return

    def _silent_arec_source(self):
        with silence_absl_errors():
            return wrap_grain_source(prop.fget(self))

    mix_mod.Arec.source = property(_silent_arec_source)


__all__ = [
    "SilentDataSource",
    "import_jax_cpu_safe",
    "install_absl_filter",
    "patch_arec_source",
    "quiet_jax_xla_bridge",
    "silence_absl_errors",
    "wrap_grain_source",
]
