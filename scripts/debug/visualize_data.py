"""Debug utility for visualizing Grain datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
import tyro
import wandb

from crossformer import cn
from crossformer.data.grain import builders, pipelines
from crossformer.data.grain.arec.arec import unpack_record

logger = logging.getLogger(__name__)


class _DecodedArrayRecord:
    """Thin wrapper decoding ArrayRecord records into Python dicts."""

    def __init__(self, shards: Iterable[Path]):
        self._ds = ArrayRecordDataSource([str(p) for p in sorted(shards)])

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int):
        return unpack_record(self._ds[index])


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


def _standardize_language(traj: dict) -> dict:
    traj = dict(traj)
    task = traj.get("task", {})
    if "language_instruction" in task and "language_instruction" not in traj:
        traj["language_instruction"] = task["language_instruction"]
    return traj


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
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


@dataclass
class Args:
    cfg: cn.Train
    arrayrecord_path: Path
    dataset_name: str | None = None
    log_every: int = 20
    fps: int = 8


def main(args: Args) -> None:
    cfg = args.cfg
    dataset_name = args.dataset_name or args.arrayrecord_path.name

    shards = sorted(args.arrayrecord_path.glob("*.arrayrecord"))
    if not shards:
        raise FileNotFoundError(f"No ArrayRecord shards found in {args.arrayrecord_path}")
    source = _DecodedArrayRecord(shards)

    first_traj = source[0]
    mappings = _infer_observation_mappings(first_traj)
    if mappings is None:
        raise ValueError("Trajectory missing observation key")
    image_keys, depth_keys, proprio_keys, proprio_dims = mappings

    language = first_traj.get("language_instruction") or first_traj.get("task", {}).get(
        "language_instruction"
    )
    standardize_fn = _standardize_language if language is not None else None

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

    traj_kwargs = cfg.data.transform.traj.create()
    traj_kwargs["window_size"] = cfg.window_size or traj_kwargs.get("window_size", 1)

    dataset = pipelines.make_single_dataset(
        dataset_config,
        train=False,
        traj_transform_kwargs=traj_kwargs,
        shuffle_buffer_size=1,
        seed=cfg.seed,
    )

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
    for step, frame in enumerate(dataset.dataset):
        stacked = _stack_views(frame, camera_views)
        if stacked is None:
            continue
        video, panels = stacked
        language = frame.get("task", {}).get("language_instruction")
        if isinstance(language, np.ndarray):
            language = language.item() if language.ndim == 0 else language[0]
        if step % args.log_every == 0:
            preview = _render_preview(panels, camera_views)
            wandb.log(
                {
                    f"videos/{dataset_name}": wandb.Video(
                        video.transpose(0, 3, 1, 2), fps=args.fps, format="gif"
                    ),
                    f"preview/{dataset_name}": wandb.Image(preview),
                    f"language/{dataset_name}": language or "",
                },
                step=step,
            )
            logged += 1
        processed += 1

    logger.info("Processed %d frames, logged %d videos", processed, logged)
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
