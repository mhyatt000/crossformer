from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Iterator, Literal

import cv2
import grain
from grain._src.python.dataset.transformations.flatmap import FlatMapIterDataset
import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
import zarr

from crossformer.data.arec.arec import ArrayRecordBuilder
from crossformer.data.grain.map import flatmap
from crossformer.data.utils.trajectory import binarize_gripper_actions as binarize
from crossformer.data.utils.trajectory import scan_noop
from crossformer.utils.spec import spec
from crossformer.utils.tree import flat, unflat


@dataclass
class Config:
    path: Path = Path.home() / "2026-04-03_2241"
    mode: Literal["preview", "build"] = "preview"
    unit: Literal["step", "episode"] = "episode"
    max_depth: int = 10
    shuffle: bool = False
    seed: int = 0
    preview: int = 1  # n preview steps
    missing: Literal["skip", "none", "error"] = "skip"
    threshold: float = 1e-3

    shard_size: int = 1000
    builder: ArrayRecordBuilder = field(init=False)

    name: str = "zarr_steps"
    version: str = "0.0.1"
    branch: str = "main"

    def build(self, fn) -> None:
        print(self)
        self.builder = ArrayRecordBuilder(
            name=self.name,
            version=self.version,
            branch=self.branch,
            shard_size=self.shard_size,
        )
        print(self.builder.root)
        self.builder.prepare(fn)


@dataclass(frozen=True)
class EpisodeInfo:
    key: str
    episode_idx: int
    length: int
    n_valid: int
    start_idx: int


@dataclass(frozen=True)
class StepInfo:
    episode_key: str
    episode_idx: int
    step_idx: int


META_KEYS = {"format", "stamp_ns"}


class ZarrLoader:
    def __init__(
        self,
        path: str | Path,
        unit: Literal["step", "episode"] = "step",
        shuffle: bool = False,
        seed: int | None = None,
        missing: Literal["skip", "none", "error"] = "skip",
    ):
        self.path = Path(path).expanduser()
        self.root = zarr.open(self.path, mode="r")
        self.unit = unit
        self.shuffle = shuffle
        self.seed = seed
        self.missing = missing

        self._episode_keys = self._find_episode_keys()
        self._keys = self._collect_topic_keys()
        self._missing_counts: dict[str, int] = {}
        self._episodes, self._steps = self._build_index()
        self._len = len(self._steps)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    @property
    def episode_keys(self) -> tuple[str, ...]:
        return tuple(ep.key for ep in self._episodes)

    @property
    def missing_counts(self) -> dict[str, int]:
        return dict(self._missing_counts)

    def __len__(self) -> int:
        if self.unit == "episode":
            return self.n_episodes
        return self._len

    def __iter__(self) -> Iterator[dict[str, Any] | list[dict[str, Any]]]:
        if self.unit == "episode":
            yield from self.iter_episodes()
            return
        yield from self.iter_steps()

    def __getitem__(self, idx: int) -> dict[str, Any] | list[dict[str, Any]]:
        if self.unit == "episode":
            return self.get_episode(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        step = self._steps[idx]
        return self._get_step(step, idx)

    def get_episode(self, idx: int) -> list[dict[str, Any]]:
        if idx < 0:
            idx += self.n_episodes
        if idx < 0 or idx >= self.n_episodes:
            raise IndexError(idx)
        ep = self._episodes[idx]
        return [
            self._get_step(StepInfo(ep.key, ep.episode_idx, step_idx), ep.start_idx + step_idx)
            for step_idx in range(ep.n_valid)
        ]

    def iter_steps(
        self,
        shuffle: bool | None = None,
        seed: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        use_shuffle = self.shuffle if shuffle is None else shuffle
        order = list(range(len(self)))
        if use_shuffle:
            rng = random.Random(self.seed if seed is None else seed)
            rng.shuffle(order)
        for idx in order:
            yield self[idx]

    def iter_episodes(self) -> Iterator[list[dict[str, Any]]]:
        for idx in range(self.n_episodes):
            yield self.get_episode(idx)

    def _find_episode_keys(self) -> tuple[str, ...]:
        keys = sorted(self.root.keys())
        if not keys:
            raise ValueError(f"empty zarr dataset: {self.path}")
        bad = [k for k in keys if not k.startswith("episode_")]
        if bad:
            raise ValueError(f"root keys must be episode_*; got {bad}")
        return tuple(keys)

    def _collect_topic_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for ep in self._episode_keys:
            keys.update(self.root[ep]["topics"].keys())
        return tuple(sorted(keys))

    def _build_index(self) -> tuple[tuple[EpisodeInfo, ...], tuple[StepInfo, ...]]:
        episodes: list[EpisodeInfo] = []
        steps: list[StepInfo] = []
        start_idx = 0
        for ep_key in self._episode_keys:
            ep = self.root[ep_key]
            n_steps = int(ep["steps"]["timestamp_ns"].shape[0])
            episode_idx = int(ep_key.removeprefix("episode_"))
            lengths = self._topic_field_lengths(ep["topics"])
            min_len = min([n_steps, *lengths.values()])
            if self.missing == "error":
                bad = [name for name, n in lengths.items() if n != n_steps]
                if bad:
                    miss = ", ".join(sorted(bad)[:4])
                    raise ValueError(f"missing topic fields in {ep_key}: {miss}")
            if self.missing == "skip":
                for name, n in lengths.items():
                    if n < n_steps:
                        self._missing_counts[name] = self._missing_counts.get(name, 0) + (n_steps - n)
                n_valid = min_len
            else:
                n_valid = n_steps
            steps.extend(StepInfo(ep_key, episode_idx, step_idx) for step_idx in range(n_valid))
            episodes.append(
                EpisodeInfo(key=ep_key, episode_idx=episode_idx, length=n_steps, n_valid=n_valid, start_idx=start_idx)
            )
            start_idx += n_valid
        return tuple(episodes), tuple(steps)

    def _topic_field_lengths(self, topics: zarr.Group) -> dict[str, int]:
        out: dict[str, int] = {}
        for topic_key in sorted(topics.keys()):
            topic = topics[topic_key]
            for name in sorted(topic.keys()):
                out[f"{topic_key}.{name}"] = int(topic[name].shape[0])
        return out

    def _get_step(self, step: StepInfo, global_idx: int) -> dict[str, Any]:
        ep = self.root[step.episode_key]
        steps = ep["steps"]
        topics = ep["topics"]
        return {
            "episode": step.episode_key,
            "episode_idx": step.episode_idx,
            "step": step.step_idx,
            "global_step": global_idx,
            "timestamp_ns": int(steps["timestamp_ns"][step.step_idx]),
            "topics": {
                topic: self._read_topic_step(topic, topics[topic], step.step_idx) for topic in sorted(topics.keys())
            },
        }

    def _read_topic_step(self, topic_key: str, topic: zarr.Group, step_idx: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in sorted(topic.keys()):
            arr = topic[name]
            out[name] = arr[step_idx] if step_idx < arr.shape[0] else None
        return self._decode_topic_step(topic_key, out)

    def _unwrap_scalar(self, x: Any) -> Any:
        while isinstance(x, np.ndarray) and x.ndim == 0:
            x = x.item()
        return x

    def _decode_topic_step(self, topic_key: str, step: dict[str, Any]) -> dict[str, Any]:
        if "data" not in step:
            return step
        if topic_key.endswith("__compressed"):
            return self._decode_compressed_image(step)
        data = step["data"]
        if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[-1] == 2:
            return self._decode_yuyv_image(step)
        return step

    def _decode_compressed_image(self, step: dict[str, Any]) -> dict[str, Any]:
        raw = self._unwrap_scalar(step["data"])
        buf = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        encoded = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("failed to decode compressed image")
        fmt = str(self._unwrap_scalar(step.get("format", "")))
        if "bgr8" in fmt or "rgb8" not in fmt:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        step["data"] = img
        step["format"] = "rgb8"
        return step

    def _decode_yuyv_image(self, step: dict[str, Any]) -> dict[str, Any]:
        img = cv2.cvtColor(np.asarray(step["data"]), cv2.COLOR_YUV2RGB_YUY2)
        step["data"] = img
        step["format"] = "rgb8"
        return step


def print_tree(group: zarr.Group, indent: int = 0, max_depth: int = 10) -> None:
    if indent > max_depth:
        return
    prefix = "  " * indent
    for key in sorted(group.keys()):
        item = group[key]
        if isinstance(item, zarr.Group):
            print(f"{prefix}{key}/")
            print_tree(item, indent + 1, max_depth)
        else:
            print(f"{prefix}{key}: shape={item.shape} dtype={item.dtype}")


def topic_data_only(x: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in x.items():
        if key in META_KEYS:
            continue
        if isinstance(value, dict):
            out[key] = topic_data_only(value)
        else:
            out[key] = value
    return out


def set_nested(x: dict[str, Any], path: str, value: Any) -> None:
    cur = x
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def map_topic_key(key: str) -> str:
    return jax.tree.map(lambda x: x.replace("__", "."), key, is_leaf=lambda x: isinstance(x, str))


def topic_batch(step_topics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in step_topics.items():
        set_nested(out, map_topic_key(key), restructure(key, topic_data_only(value)))
    return out


def restructure(key: str, value: dict[str, Any]) -> dict[str, Any]:
    if key == "xgym__gripper":
        return parse_gripper(value)
    if key == "xarm__robot_states":
        return parse_robot_state(value)
    return value


def parse_json(raw: Any) -> Any:
    if isinstance(raw, np.ndarray):
        return [json.loads(str(x)) for x in raw]
    return json.loads(str(raw))


def parse_gripper(value: dict[str, Any]) -> dict[str, Any]:
    raw = value.get("json")
    if raw is None:
        return value
    msg = parse_json(raw)
    data = [x.get("data", []) for x in msg] if isinstance(msg, list) else msg.get("data", [])
    return {"position": np.asarray(data, dtype=np.float32)}


def parse_robot_state(value: dict[str, Any]) -> dict[str, Any]:
    raw = value.get("json")
    if raw is None:
        return value
    msg = parse_json(raw)
    if isinstance(msg, list):
        out: dict[str, Any] = {}
        keys = ("angle", "pose", "offset", "state", "mode", "cmdnum", "mt_brake", "mt_able", "err", "warn")
        for k in keys:
            vals = [x[k] for x in msg if k in x]
            if vals:
                dtype = np.float32 if k in {"angle", "pose", "offset"} else None
                out[k] = np.asarray(vals, dtype=dtype)
        return out
    out: dict[str, Any] = {}
    if "angle" in msg:
        out["angle"] = np.asarray(msg["angle"], dtype=np.float32)
    if "pose" in msg:
        out["pose"] = np.asarray(msg["pose"], dtype=np.float32)
    if "offset" in msg:
        out["offset"] = np.asarray(msg["offset"], dtype=np.float32)
    for k in ("state", "mode", "cmdnum", "mt_brake", "mt_able", "err", "warn"):
        if k in msg:
            out[k] = np.asarray(msg[k])
    return out


def standardize(step: dict[str, Any], threshold: float = 1e-3) -> dict[str, Any]:
    step = {
        "observation": topic_batch(step["topics"]),
        "info": {
            "id": {
                # "episode": step["episode"], # this is episode path
                "episode": np.asarray(step["episode_idx"]),
                "step": np.asarray(step["step"]),
                "global": np.asarray(step["global_step"]),
                # "timestamp_ns": np.asarray(step["timestamp_ns"]),
            },
        },
    }
    out = {}
    out["info"] = step["info"]
    step = flat(step)
    out["observation.image.low"] = step["observation.cam.low.image_raw.data"]
    out["observation.image.side"] = step["observation.cam.side.image_raw.data"]
    out["observation.image.wrist"] = step["observation.camera.camera.color.image_raw.compressed.data"]

    pose = step["observation.xarm.robot_states.pose"]
    out["observation.proprio.position"] = pose[..., :3] / 1e3
    out["observation.proprio.orientation"] = pose[..., 3:]
    out["observation.proprio.gripper"] = step["observation.xgym.gripper.position"]
    out["observation.proprio.joints"] = step["observation.xarm.joint_states.position"]

    out["observation.proprio.gripper"] = np.asarray(
        binarize(jnp.asarray(out["observation.proprio.gripper"]), open=0.95, close=0.4)
    )

    pos = np.concatenate((out["observation.proprio.position"], out["observation.proprio.gripper"]), axis=-1)
    noops = np.asarray(scan_noop(jnp.asarray(pos), threshold=threshold))
    jpos = np.concatenate((out["observation.proprio.joints"], out["observation.proprio.gripper"]), axis=-1)
    jnoop = np.asarray(scan_noop(jnp.asarray(jpos), threshold=threshold))
    mask = np.logical_and(~noops, ~jnoop)
    out = jax.tree.map(lambda x: x[mask], out)
    n = len(out["info"]["id"]["step"])
    out["info"]["len"] = np.full((n,), n)
    out = unflat(out)
    return out


def main(cfg: Config) -> None:
    loader = ZarrLoader(cfg.path, unit=cfg.unit, shuffle=cfg.shuffle, seed=cfg.seed, missing=cfg.missing)
    ds = grain.MapDataset.source(loader)
    if cfg.unit == "episode":
        ds = ds.map(lambda X: jax.tree.map(lambda *xs: np.stack(xs), *X))
    ds = ds.map(lambda x: standardize(x, threshold=cfg.threshold))

    print(f"Root: {loader.path}")
    print_tree(loader.root, max_depth=cfg.max_depth)
    print()
    print(f"n_episodes={loader.n_episodes}")
    print(f"len={len(loader)}")
    print(f"keys={loader.keys}")
    if loader.missing_counts:
        print(f"missing_policy={loader.missing}")
        print(f"dropped_steps={sum(ep.length for ep in loader._episodes) - len(loader)}")
        print(f"missing_fields={loader.missing_counts}")

    if cfg.mode == "preview":
        n = min(cfg.preview, len(ds))
        for i in tqdm(range(n)):
            print(spec(ds[i]))
        return

    ckpt = list(tqdm(ds, total=len(ds)))
    n_step = sum(len(x["info"]["id"]["step"]) for x in tqdm(ckpt, total=len(ckpt)))

    ds = FlatMapIterDataset(
        grain.MapDataset.source(ckpt),
        transform=flatmap.UnpackFlatMap(key="info.id.step", use_np=True),
    )
    ds = grain.MapDataset.source(list(ds))

    bar = tqdm(total=n_step, desc="Building dataset")

    def build_progress(data: dict) -> dict:
        bar.update(1)
        return data

    ds = ds.map(build_progress)

    def build_fn() -> Iterator[dict[str, Any]]:
        yield from ds

    cfg.build(build_fn)
    bar.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
