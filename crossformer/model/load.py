from __future__ import annotations

from enum import StrEnum
import os
from pathlib import Path
import urllib.request


class Variant(StrEnum):
    tips_oss_g14_highres = "tips_oss_g14_highres"
    tips_oss_g14_lowres = "tips_oss_g14_lowres"
    tips_oss_so400m14_highres_largetext_distilled = "tips_oss_so400m14_highres_largetext_distilled"
    tips_oss_l14_highres_distilled = "tips_oss_l14_highres_distilled"
    tips_oss_b14_highres_distilled = "tips_oss_b14_highres_distilled"
    tips_oss_s14_highres_distilled = "tips_oss_s14_highres_distilled"
    tips_v2_g14 = "tips_v2_g14"
    tips_v2_so14 = "tips_v2_so14"
    tips_v2_l14 = "tips_v2_l14"
    tips_v2_b14 = "tips_v2_b14"


def variant_name(variant: Variant | str) -> str:
    return variant.value if isinstance(variant, Variant) else variant


def cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
    return base / "tips" / "checkpoints"


def checkpoint_url(variant: Variant | str) -> str:
    name = variant_name(variant)
    version = "v2_0" if name.startswith("tips_v2_") else "v1_0"
    return f"https://storage.googleapis.com/tips_data/{version}/checkpoints/scenic/{name}_vision.npz"


def default_checkpoint_path(variant: Variant | str) -> Path:
    return cache_dir() / f"{variant_name(variant)}_vision.npz"


def resolve_checkpoint_path(
    variant: Variant | str,
    checkpoint_path: str | Path | None,
) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path).expanduser()

    name = variant_name(variant)
    path = default_checkpoint_path(name)
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = checkpoint_url(name)
    print(f"Downloading {name} JAX checkpoint to {path}")
    urllib.request.urlretrieve(url, path)
    return path
