from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterator, Literal

import cv2
import numpy as np
from rich import print
from tqdm import tqdm
import tyro
import zarr

from crossformer.utils.spec import spec


@dataclass
class Config:
    path: Path = Path.home() / "2026-04-03_2241"
    max_depth: int = 10
    shuffle: bool = False
    seed: int = 0
    preview: int = 1  # n preview steps
    missing: Literal["skip", "none", "error"] = "skip"


@dataclass(frozen=True)
class EpisodeInfo:
    key: str
    length: int
    n_valid: int


@dataclass(frozen=True)
class StepInfo:
    episode_key: str
    episode_idx: int
    step_idx: int


class ZarrLoader:
    def __init__(
        self,
        path: str | Path,
        shuffle: bool = False,
        seed: int | None = None,
        missing: Literal["skip", "none", "error"] = "skip",
    ):
        self.path = Path(path).expanduser()
        self.root = zarr.open(self.path, mode="r")
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
        return self._len

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.iter_steps()

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        step = self._steps[idx]
        return self._get_step(step, idx)

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
            episodes.append(EpisodeInfo(key=ep_key, length=n_steps, n_valid=n_valid))
        return tuple(episodes), tuple(steps)

    def _topic_field_lengths(self, topics: zarr.Group) -> dict[str, int]:
        out: dict[str, int] = {}
        for topic_key in sorted(topics.keys()):
            topic = topics[topic_key]
            for field in sorted(topic.keys()):
                out[f"{topic_key}.{field}"] = int(topic[field].shape[0])
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
        for field in sorted(topic.keys()):
            arr = topic[field]
            out[field] = arr[step_idx] if step_idx < arr.shape[0] else None
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


def main(cfg: Config) -> None:
    loader = ZarrLoader(cfg.path, shuffle=cfg.shuffle, seed=cfg.seed, missing=cfg.missing)
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

    for i, step in tqdm(enumerate(loader.iter_steps())):
        if i >= cfg.preview:
            break
        if i == 0:
            print(
                {
                    "episode": step["episode"],
                    "step": step["step"],
                    "global_step": step["global_step"],
                    "timestamp_ns": step["timestamp_ns"],
                    "topic_keys": tuple(step["topics"].keys()),
                }
            )
            print(spec(step["topics"]))


if __name__ == "__main__":
    main(tyro.cli(Config))
