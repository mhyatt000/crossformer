"""Debug utility for visualizing Grain datasets."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
from typing import Literal

import flax.traverse_util as ftu
import grain
import matplotlib.pyplot as plt
import numpy as np
from rich.pretty import pprint
import tyro

from crossformer import cn
from crossformer.data.grain import builders, pipelines
from crossformer.data.grain.datasets import _DecodedArrayRecord, _DropKeyDataset, _EpisodeDataset, drop
from crossformer.data.grain.map.window import FlatMapDataset, WindowedFlatMap
from crossformer.data.grain.util.remap import _remap_lang
from crossformer.utils.spec import spec
import wandb

log = logging.getLogger(__name__)


def _infer_observation_mappings(traj: dict) -> tuple[dict, dict, dict, dict] | None:
    obs = traj.get("observation")
    if obs is None:
        return None

    image_keys = {}
    depth_keys = {}
    proprio_keys = {}
    proprio_dims = {}

    for key, value in obs.items():
        if key.startswith("image_"):
            image_keys[key.removeprefix("image_")] = key
        elif key.startswith("depth_"):
            depth_keys[key.removeprefix("depth_")] = key
        elif key.startswith("proprio_"):
            proprio_keys[key.removeprefix("proprio_")] = key
            proprio_dims[key.removeprefix("proprio_")] = int(np.shape(value)[-1])

    return image_keys, depth_keys, proprio_keys or None, proprio_dims or None


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

    def __post_init__(self):
        logging.basicConfig(level=self.log_level.upper(), force=True)
        log.info(f"Logging level set to {self.log_level.upper()}")
        return super().__post_init__()


@dataclass
class DataConfig:
    # reader
    path: Path


def main(cfg: Config) -> None:
    dataset_name = cfg.dataset_name or cfg.arec_path.name

    shards = sorted(cfg.arec_path.glob("*.arrayrecord"))
    if not shards:
        raise FileNotFoundError(f"No ArrayRecord shards found in {cfg.arec_path}")

    ds = _DecodedArrayRecord(shards[:5])
    ds = _EpisodeDataset(ds)
    # it = iter(ds)
    # ds = [next(it) for _ in range(2)]
    w = WindowedFlatMap(size=5, stride=1)
    L = sum([w.len(_l) for _l in ds.lengths()])
    # ds = FlatMapDataset(ds, w, L=L)
    # ds = PrefetchWrapper(ds, transform=None)
    ds = grain.MapDataset.source(ds)

    # with PrefetchWrapper(ds, transform=None) as loader:
    pprint(spec(next(iter(ds))))
    quit()

    ds = _DropKeyDataset(
        ds,
        drop_keys=[
            "discount",
            "is_first",
            "is_terminal",
            "reward",
        ],
    )

    def flat(tree):
        return {".".join(k): v for k, v in ftu.flatten_dict(tree).items()}

    def unflat(tree):
        return ftu.unflatten_dict({tuple(k.split(".")): v for k, v in tree.items()})

    ds = grain.MapDataset.source(ds)
    ds = ds.map(flat)
    ds = ds.map(
        partial(
            drop,
            keys=[
                "discount",
                "is_first",
                "is_terminal",
                "reward",
            ],
        )
    )

    ds = FlatMapDataset(ds, WindowedFlatMap(size=5, stride=1))
    pprint(spec(next(iter(ds))))

    ds = grain.IterDataset(ds)
    ds = ds.map(unflat)

    quit()

    first_traj = source[0]
    mappings = _infer_observation_mappings(first_traj)
    if mappings is None:
        raise ValueError("Trajectory missing observation key")
    image_keys, depth_keys, proprio_keys, proprio_dims = mappings

    language = first_traj.get("language_instruction")
    standardize_fn = _remap_lang if language is not None else None

    dataset_config = builders.GrainDatasetConfig(
        name=dataset_name,
        source=source,
        standardize_fn=standardize_fn,
        image_obs_keys=image_keys,
        depth_obs_keys=depth_keys,
        proprio_obs_keys=proprio_keys,
        proprio_obs_dims=proprio_dims,
        language_key="language_instruction" if language is not None else None,
        skip_norm_keys=cfg.data.transform.skip_norm_keys,
    )

    traj_kwargs = cfg.data.traj.create(with_head_to_dataset=False)
    traj_kwargs["window_size"] = cfg.window_size or traj_kwargs.get("window_size", 1)
    traj_kwargs.pop("task_augment_strategy")
    traj_kwargs.pop("task_augment_kwargs")

    dataset = pipelines.make_single_dataset(
        dataset_config,
        train=False,
        traj_transform_kwargs=traj_kwargs,
        shuffle_buffer_size=1,
        seed=cfg.seed,
    )

    pprint(spec(next(iter(dataset.dataset))))
    quit()

    wandb_mode = cfg.wandb.mode(cfg.debug)
    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        entity=cfg.wandb.entity,
        mode=wandb_mode,
        config={"dataset": dataset_name, "window_size": traj_kwargs["window_size"]},
        name=f"visualize_{dataset_name}",
    )

    camera_views = list(image_keys.keys()) or cfg.data.reader.load_camera_views
    logged = 0
    processed = 0

    logger.info("Processed %d frames, logged %d videos", processed, logged)
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
