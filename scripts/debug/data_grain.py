"""Debug utility for visualizing Grain datasets."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
import tyro

from crossformer import cn
from crossformer.data.grain import pipelines
from crossformer.data.grain.map.window import FlatMapDataset, WindowedFlatMap, WindowFlatDataset  # noqa
from crossformer.utils.spec import spec

log = logging.getLogger(__name__)


def _infer_observation_mappings(tree: dict) -> tuple[dict, dict, dict, dict] | None:
    obs = tree.get("observation", {})
    image_keys = set(obs.get("image", {}))
    depth_keys = set(obs.get("depth", {}))

    proprio = {k: v[-1] for k, v in spec(obs.get("proprio", {})).items()}
    # proprio_keys, proprio_dims = zip(*proprio.items())

    return image_keys, depth_keys, proprio


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.uint8)
    arr = arr.copy()
    arr[~finite] = 0
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v > 1e-6:
        arr = (arr - min_v) / (max_v - min_v)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def _stack_views(frame: dict, views: list[str]) -> tuple[np.ndarray, list[np.ndarray]] | None:
    panels = []
    for view in views:
        key = f"image_{view}"
        imgs = frame.get("observation", {}).get(key)
        if imgs is None:
            continue
        imgs = np.asarray(imgs)
        if imgs.ndim == 3:
            imgs = imgs[None, ...]
        imgs = _to_uint8(imgs)
        panels.append(imgs)
    if not panels:
        return None
    videos = [np.concatenate(frames, axis=1) for frames in zip(*panels)]
    return np.stack(videos), panels


def _render_preview(panels: list[np.ndarray], views: list[str]) -> np.ndarray:
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    for ax, view, panel in zip(axes, views, panels):
        ax.imshow(panel[0])
        ax.set_title(view)
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape((fig.canvas.get_width_height()[::-1], 3))
    plt.close(fig)
    return image


@dataclass
class Config(cn.Train):
    arec_path: Path | None = None
    dataset_name: str | None = None
    log_level: Literal["debug", "info", "warning", "error"] = "info"  # logging verbosity
    log_every: int = 20
    fps: int = 8
    drop_observation_keys: tuple[str, ...] = ()
    recompute: bool = False  # recompute data stats? y/n

    def __post_init__(self):
        logging.basicConfig(level=self.log_level.upper(), force=True)
        log.info(f"Logging level set to {self.log_level.upper()}")
        return super().__post_init__()


@dataclass
class DataConfig:
    # reader
    path: Path


def main(cfg: Config) -> None:
    _ds, dataset_config, tfconfig = pipelines.make_data_source(cfg)
    dataset = pipelines.make_single_dataset(
        dataset_config,
        train=True,
        tfconfig=tfconfig,
        shuffle_buffer_size=1,
        seed=cfg.seed,
    )

    pprint(dataset)
    pprint(spec(next(iter(dataset.dataset))))
    dsit = iter(dataset.dataset)
    for _ in tqdm(range(int(1e4))):
        x = next(dsit)


if __name__ == "__main__":
    main(tyro.cli(Config))
