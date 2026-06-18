from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import grain
import jax
import numpy as np
from rich import print
from tqdm import tqdm
import tyro

from crossformer.data.grain.loader import _apply_fd_limit
from crossformer.data.mcap import McapLoader
from crossformer.utils.spec import diff, SimpleSpec, spec
from crossformer.utils.tree import flat


@dataclass
class Config:
    path: Path
    preview: int | None = None  # n preview
    recursive: bool = True
    max_messages_per_topic: int | None = None
    read_threads: int = 32
    prefetch_buffer_size: int = 32

    verbose: bool = False
    vbar: bool = False  # show message-level progress bars

    mp: int = 4
    mp_buf: int = 4  # per worker buffer size


def first_spec_match():
    """Compare each item spec with the first."""
    first = None

    def match(item) -> tuple[bool, dict]:
        nonlocal first
        step = jax.tree.map(lambda x: x[0] if getattr(x, "ndim", 0) else x, item)
        current = {key: value for key, value in spec(flat(step)).items() if isinstance(value, SimpleSpec)}
        if first is None:
            first = current
            return True, {}
        delta = diff(first, current)
        return (True, {}) if not any(delta.values()) else (False, delta)

    return match


def _constant(x, name: str):
    x = np.asarray(x)
    if not len(x) or not np.all(x == x[0]):
        raise ValueError(f"RawImage {name} must be constant")
    return x[0].item()


def map_topic_payloads(tree: dict) -> dict:
    """Replace known topic payloads with their useful data."""
    out = {}
    for key, value in tree.items():
        if not isinstance(value, dict):
            out[key] = value
            continue

        schema = value.get("schema")
        if schema == "foxglove.JointStates":
            out[key] = value["data"]["joints"]["position"]
            continue
        if schema == "foxglove.Pose":
            out[key] = value["data"]
            continue
        if schema == "xclients.Gripper":
            out[key] = value["data"]["norm"]
            continue
        if schema != "foxglove.RawImage":
            out[key] = map_topic_payloads(value)
            continue

        data = value["data"]
        h = _constant(data["height"], "height")
        w = _constant(data["width"], "width")
        step = _constant(data["step"], "step")
        if step % w:
            raise ValueError(f"RawImage step={step} is not divisible by width={w}")
        c = step // w
        images = data["data"]
        if images.shape[1] != h * step:
            raise ValueError(f"RawImage data width={images.shape[1]} does not match height x step={h * step}")
        images = images.reshape(len(images), h, w, c)
        if c == 2:
            images = np.stack([cv2.cvtColor(image, cv2.COLOR_YUV2RGB_YUY2) for image in images])
        out[key] = images
    return out


def main(cfg: Config) -> None:
    loader = McapLoader(
        cfg.path,
        recursive=cfg.recursive,
        max_messages_per_topic=cfg.max_messages_per_topic,
    )
    print(f"Root: {loader.path}")
    print(f"n_episodes={len(loader)}")

    n = max(0, min(cfg.preview, len(loader))) if cfg.preview else len(loader)
    ds = loader.iter_dataset(
        read_threads=cfg.read_threads,
        prefetch_buffer_size=min(cfg.prefetch_buffer_size, n),
        stop=n,
        show_message_progress=cfg.vbar,
    )
    ds = ds.map(map_topic_payloads)

    lim = _apply_fd_limit(512**2)
    ds = ds.mp_prefetch(
        grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_buf),
    )
    # ds = ds.map(first_spec_match())

    for i, x in enumerate(tqdm(ds, total=n, desc="Loading episodes", position=0)):
        if cfg.verbose:
            print(f"\n[bold]episode={i}[/bold]")
        print(spec(x))


if __name__ == "__main__":
    main(tyro.cli(Config))
