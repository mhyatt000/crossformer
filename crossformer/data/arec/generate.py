from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
from typing import Any, ClassVar
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from crossformer.data.utils.trajectory import binarize_gripper_actions as binarize
from crossformer.data.utils.trajectory import scan_noop
from crossformer.utils.io import memmap

log = logging.getLogger(__name__)

warnings.simplefilter(action="ignore", category=FutureWarning)

threshold = 1e-3


class Builder:
    """DatasetBuilder Base"""

    # VERSION = tfds.core.Version("3.0.0")
    RELEASE_NOTES: ClassVar = {
        "1.0.0": "Initial release.",
        "2.0.0": "more data and overhead cam",
        "3.0.0": "relocated setup",
        "4.0.0": "50hz data",
    }

    def __init__(self, workers=32):
        # super().__init__()
        self.threshold = 1e-3
        self.workers = workers

    def build(self):
        """Define data splits."""

        # if not self.root:
        # print(self.root)
        # self.root = Path.home() / f"{self.name}:{ str(self.VERSION)[0]}"
        self.files = list(self.root.rglob("*.dat"))
        if self.limit:
            self.files = self.files[: self.limit]
        return self._generate_examples(self.files)

    def _parse_example(self, path):
        log.debug(path)

        try:
            info, ep = memmap.read(path)
            cams = [k for k in info["schema"] if "camera" in k]
            if len(cams) < 2:
                raise ValueError(f"Not enough cameras {cams}")
        except Exception as e:
            log.error("Error reading %s", path)
            log.error("%s", e)
            return None

        n = len(ep["time"])

        ### cleanup and remap keys
        ep.pop("time")
        ep.pop("gello_joints")

        ep["robot"] = {}
        ep["robot"]["joints"] = ep.pop("xarm_joints")
        ep["robot"]["position"] = ep.pop("xarm_pose")
        ep["robot"]["gripper"] = ep.pop("xarm_gripper")
        # ep['robot'] = {'joints': joints, 'position': np.concatenate((pose, grip), axis=1)}

        if "/xgym/camera/worm" not in ep or "/xgym/camera/low" in ep:
            ep["/xgym/camera/worm"] = ep.pop("/xgym/camera/low")
        if "/xgym/camera/wrist" not in ep:
            ep["/xgym/camera/wrist"] = ep.pop("/xgym/camera/rs")

        ep["image"] = {k: ep.pop(f"/xgym/camera/{k}") for k in ["worm", "side", "wrist"]}

        ### scale and binarize
        ep["robot"]["gripper"] /= 850
        ep["robot"]["position"][:, :3] /= 1e3
        _binarize = partial(binarize, open=0.95, close=0.4)  # doesnt fully close
        ep["robot"]["gripper"] = np.array(_binarize(jnp.array(ep["robot"]["gripper"])))

        ### filter noop cartesian
        pos = np.concatenate((ep["robot"]["position"], ep["robot"]["gripper"]), axis=1)
        noops = np.array(scan_noop(jnp.array(pos), threshold=self.threshold))
        mask = ~noops
        # filter noop joints
        jpos = np.concatenate([ep["robot"]["joints"], ep["robot"]["gripper"]], axis=1)
        jnoop = np.array(scan_noop(jnp.array(jpos), threshold=self.threshold))
        jmask = ~jnoop
        mask = np.logical_and(mask, jmask)

        log.debug(f"Kept {mask.sum()} of {n} steps")

        ep = jax.tree.map(select := lambda x: x[mask], ep)

        ### calculate action
        action = jax.tree.map(lambda x: x[1:] - x[:-1], ep["robot"])  # pose and joint action
        action["gripper"] = ep["robot"]["gripper"][1:]  # gripper is absolute
        ep = jax.tree.map(lambda x: x[:-1], ep)
        # ep["action"] = action # action is not an observation

        ### final remaps
        ep["proprio"] = ep.pop("robot")

        geti = lambda x, i: jax.tree.map(lambda y: y[i], x)
        episode = [
            {
                "observation": geti(ep, i),
                "action": geti(action, i),
                "discount": 1.0,
                "reward": float(i == (len(ep) - 1)),
                "is_first": i == 0,
                "is_last": i == (len(ep) - 1),
                "is_terminal": i == (len(ep) - 1),
                "language_instruction": self.task,
                "language_embedding": self.lang,
            }
            for i in range(len(ep["proprio"]["position"]))
        ]

        # if you want to skip an example for whatever reason, simply return None
        sample = {"steps": episode, "episode_metadata": {}}
        id = f"{path.parent.name}_{path.stem}"
        return episode if len(episode) > 0 else None
        return id, sample

    def _generate_examples(self, ds) -> Iterator[tuple[str, Any]]:
        """Generator of examples for each split."""

        self.taskfile = next(self.root.glob("*.npy"), None)
        if self.taskfile is None:
            raise FileNotFoundError(f"No task embedding .npy file found under {self.root}")
        self.task = self.taskfile.stem.replace("_", " ")
        self.lang = np.load(self.taskfile)

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for result in ex.map(self._parse_example, ds):
                if result is not None:
                    yield result
