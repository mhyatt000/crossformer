from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from crossformer.data.utils.trajectory import binarize_gripper_actions as binarize
from crossformer.data.utils.trajectory import scan_noop
from crossformer.utils.io import memmap

log = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class Builder:
    """DatasetBuilder Base"""

    root: Path
    threshold: float = 1e-3
    workers: int = 32

    def build(self):
        """Define data splits."""

        self.files = list(self.root.rglob("*.dat"))
        if self.limit:
            self.files = self.files[: self.limit]
        return self._generate_examples(self.files)

    def _generate_examples(self, ds) -> Iterator[list[dict]]:
        """Generator of examples for each split."""

        self.task, self.lang = self._load_task()

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for result in ex.map(self._parse_example, ds):
                if result is not None:
                    yield result

    def _parse_example(self, path: Path):
        log.debug(path)

        try:
            _, ep = memmap.read(path)
            self._validate_cameras(ep)
            ep = self._remap_episode(ep)
            ep = self._normalize_episode(ep)
            ep = self._filter_noops(ep)
            return self._build_episode(ep)
        except Exception as e:
            log.error("Error reading %s", path)
            log.error("%s", e)
            return None

    def _validate_cameras(self, ep: dict) -> None:
        cams = [k for k in ep if "camera" in k]
        if len(cams) < 2:
            raise ValueError(f"Not enough cameras {cams}")

    def _remap_episode(self, ep: dict) -> dict:
        ep.pop("time")
        ep.pop("gello_joints")

        ep["robot"] = {
            "joints": ep.pop("xarm_joints"),
            "position": ep.pop("xarm_pose"),
            "gripper": ep.pop("xarm_gripper"),
        }

        self._remap_cameras(ep)
        ep["image"] = {k: ep.pop(f"/xgym/camera/{k}") for k in ("worm", "side", "wrist")}
        return ep

    def _remap_cameras(self, ep: dict) -> None:
        if "/xgym/camera/worm" not in ep and "/xgym/camera/low" in ep:
            ep["/xgym/camera/worm"] = ep.pop("/xgym/camera/low")
        if "/xgym/camera/wrist" not in ep:
            ep["/xgym/camera/wrist"] = ep.pop("/xgym/camera/rs")

    def _normalize_episode(self, ep: dict) -> dict:
        ep["robot"]["gripper"] /= 850
        ep["robot"]["position"][:, :3] /= 1e3
        _binarize = partial(binarize, open=0.95, close=0.4)
        ep["robot"]["gripper"] = np.array(_binarize(jnp.array(ep["robot"]["gripper"])))
        return ep

    def _filter_noops(self, ep: dict) -> dict:
        n = len(ep["robot"]["position"])
        pos = np.concatenate((ep["robot"]["position"], ep["robot"]["gripper"]), axis=1)
        pose_mask = ~np.array(scan_noop(jnp.array(pos), threshold=self.threshold))
        jpos = np.concatenate([ep["robot"]["joints"], ep["robot"]["gripper"]], axis=1)
        joint_mask = ~np.array(scan_noop(jnp.array(jpos), threshold=self.threshold))
        mask = np.logical_and(pose_mask, joint_mask)

        log.debug("Kept %s of %s steps", mask.sum(), n)
        return jax.tree.map(lambda x: x[mask], ep)

    def _build_episode(self, ep: dict):
        action = jax.tree.map(lambda x: x[1:] - x[:-1], ep["robot"])
        action["gripper"] = ep["robot"]["gripper"][1:]
        ep = jax.tree.map(lambda x: x[:-1], ep)

        ep["proprio"] = ep.pop("robot")
        size = len(ep["proprio"]["position"])
        episode = [
            {
                "observation": self._at(ep, i),
                "action": self._at(action, i),
                "discount": 1.0,
                "reward": float(i == (size - 1)),
                "is_first": i == 0,
                "is_last": i == (size - 1),
                "is_terminal": i == (size - 1),
                "language_instruction": self.task,
                "language_embedding": self.lang,
            }
            for i in range(size)
        ]
        return episode or None

    def _at(self, tree: dict, i: int):
        return jax.tree.map(lambda x: x[i], tree)

    def _load_task(self) -> tuple[str, np.ndarray]:
        taskfile = next(self.root.glob("*.npy"), None)
        if taskfile is None:
            raise FileNotFoundError(f"No task embedding .npy file found under {self.root}")
        task = taskfile.stem.replace("_", " ")
        return task, np.load(taskfile)
