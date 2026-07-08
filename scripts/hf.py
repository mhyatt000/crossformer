from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import flax.traverse_util as ftu

from crossformer.utils.spec import ModuleSpec


class HFModel:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @staticmethod
    def from_pretrained(
        repo_id: str | Path,
        *,
        cls: Callable[[dict[str, Any]], Any] | ModuleSpec | str | None = None,
        revision: str | None = None,
        config_file: str = "config.json",
        weights_file: str = "model.safetensors",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Load a JAX model and params from HF Hub or a local safetensors folder."""
        path = _resolve_path(
            repo_id,
            revision=revision,
            config_file=config_file,
            weights_file=weights_file,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
        config = _load_json(path / config_file)
        model_cls = _resolve_cls(cls) if cls is not None else _infer_cls(config)

        from safetensors.flax import load_file

        params = unflatten_params(load_file(path / weights_file))
        return model_cls(config), params


def unflatten_params(flat: dict[str, Any], sep: str | None = None) -> dict[str, Any]:
    """Unflatten safetensors params saved with slash or dot separated names."""
    if sep is None:
        sep = "/" if any("/" in k for k in flat) else "."
    return ftu.unflatten_dict({tuple(k.split(sep)): v for k, v in flat.items()})


def _resolve_path(
    repo_id: str | Path,
    *,
    revision: str | None,
    config_file: str,
    weights_file: str,
    cache_dir: str | Path | None,
    local_files_only: bool,
    token: str | bool | None,
) -> Path:
    src = str(repo_id)
    if src.startswith("file://"):
        return Path(src.removeprefix("file://")).expanduser().resolve()

    path = Path(src).expanduser()
    if path.exists():
        return path.resolve()

    if src.startswith("hf://"):
        src = src.removeprefix("hf://")
    if src.startswith("https://huggingface.co/"):
        src = _repo_id_from_url(src)

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            src,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            local_files_only=local_files_only,
            token=token,
            allow_patterns=[config_file, weights_file],
        )
    )


def _repo_id_from_url(url: str) -> str:
    rest = url.removeprefix("https://huggingface.co/").strip("/")
    parts = rest.split("/")
    if len(parts) < 2:
        raise ValueError(f"Expected a HF model URL with owner/repo, got {url!r}")
    return "/".join(parts[:2])


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _infer_cls(config: dict[str, Any]) -> Callable[[dict[str, Any]], Any]:
    for key in ("cls", "model_cls", "model_class", "model_spec", "module_spec"):
        if key in config:
            return _resolve_cls(config[key])

    if architectures := config.get("architectures"):
        return _resolve_cls(architectures[0])

    raise ValueError(
        "Could not infer model class from config.json. "
        "Pass cls=... or add a ModuleSpec under one of: "
        "cls, model_cls, model_class, model_spec, module_spec."
    )


def _resolve_cls(cls: Callable[[dict[str, Any]], Any] | ModuleSpec | str) -> Callable[[dict[str, Any]], Any]:
    if isinstance(cls, dict):
        return ModuleSpec.instantiate(cls)
    if isinstance(cls, str):
        return ModuleSpec.instantiate(ModuleSpec.create(_normalize_cls_path(cls)))
    if callable(cls):
        return cls
    raise TypeError(f"Expected callable, ModuleSpec, or import string for cls, got {type(cls)!r}")


def _normalize_cls_path(path: str) -> str:
    if ":" in path:
        return path
    if "." not in path:
        raise ValueError(f"Expected fully qualified class path, got {path!r}")
    module, name = path.rsplit(".", 1)
    return f"{module}:{name}"
