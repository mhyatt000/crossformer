from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from webpolicy.base_policy import BasePolicy

from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.base_policy import CorePolicy
from crossformer.run.wrappers import (
    BodyPartGroupWrapper,
    DtypeGuardWrapper,
    EnsemblerWrapper,
    HistoryWrapper,
    ImageResizeWrapper,
    LegacyDenormWrapper,
    ObsPaddingWrapper,
    ProprioNormWrapper,
    XFlowDenormWrapper,
)
from crossformer.utils.spec import spec
from crossformer.utils.tree.core import drop_fn


def resize(img, size):
    x = jnp.asarray(img)
    *lead, h, w, c = x.shape
    x = x.reshape((-1, h, w, c)) if lead else x[None]
    x = jax.image.resize(x, (x.shape[0], size[0], size[1], c), method="lanczos3", antialias=True)
    x = jnp.clip(jnp.rint(x), 0, 255).astype(jnp.uint8)
    x = x.reshape((*lead, size[0], size[1], c)) if lead else x[0]
    return np.asarray(x)


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = jax.tree.map(lambda *xs: np.stack(xs), *history)
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


@dataclass
class PolicyConfig:
    path: str
    task: str  # task to perform
    step: int | None = None  # step to load... defaults to latest if None

    chunk: int = 20  # action chunk size
    exp: float = 0.99  # exponential weighting for ensembler, higher means more weight on recent predictions
    warmup: bool = True  # whether to run a warmup phase to trigger compilation and stabilize predictions

    def verify(self):
        path = Path(self.path).expanduser().resolve()
        assert self.path and path.exists(), f"Model path {self.path} does not exist"
        cfg = path / "config.json"
        if not cfg.exists():
            entries = sorted(p.name for p in path.iterdir()) if path.is_dir() else []
            raise FileNotFoundError(f"Expected checkpoint config at {cfg}")
        assert self.task in TASKS, f"Unknown task '{self.task}'. Choose from: {sorted(TASKS)}"


@dataclass
class PolicyV2Config:
    """Configurable wrapper-based policy (v2)."""

    path: str
    task: str
    step: int | None = None
    head_name: str | None = None  # auto-detected if None

    dtype_guard: bool = True  # check for non-numeric leaves before model ingestion
    resize: bool = True  # resize images to checkpoint sizes
    obs_pad: bool = True  # zero-fill missing obs keys from checkpoint example
    proprio_norm: bool = True  # normalize proprio with dataset stats
    history: int = 1  # observation window horizon
    ensemble: bool = False  # exponential action ensembling
    chunk: int = 20  # action chunk size
    exp: float = 0.99  # ensembler exponential weight
    denorm: bool = True  # denormalize actions (xflow or legacy)
    warmup: bool = True

    def verify(self):
        path = Path(self.path).expanduser().resolve()
        assert self.path and path.exists(), f"Model path {self.path} does not exist"
        cfg = path / "config.json"
        if not cfg.exists():
            entries = sorted(p.name for p in path.iterdir()) if path.is_dir() else []
            raise FileNotFoundError(f"Expected checkpoint config at {cfg}")
        assert self.task in TASKS, f"Unknown task '{self.task}'. Choose from: {sorted(TASKS)}"


@dataclass
class ServerConfig:
    """Server configuration"""

    policy: PolicyConfig | PolicyV2Config
    host: str = "0.0.0.0"  # host to run on
    port: int = 8001  # port to run on


TASKS = {
    "sweep": {
        "text": " sweep beans into the dustpan",
        "dataset_name": "xgym_sweep_single",
    },
    "duck": {
        "text": "put the ducks in the bowl",
        "dataset_name": "xgym_duck_single",
    },
    "stack": {
        "text": "stack all the blocks vertically ",
        "dataset_name": "xgym_stack_single",
    },
    "lift": {
        "text": "pick up the red block",
        "dataset_name": "xgym_lift_single",
    },
    "play": {"text": "pick up any object", "dataset_name": "xgym_play_single"},
}


class Ensembler:
    def __init__(self, exp_weight: float, pred_horizon: int):
        self.exp_weight = exp_weight
        self.history = deque(maxlen=pred_horizon)

    def reset(self):
        self.history.clear()

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        self.history.append(actions)
        n = len(self.history)
        curr = np.stack([pred[i] for i, pred in zip(range(n - 1, -1, -1), self.history)])
        weights = np.exp(-self.exp_weight * np.arange(n))
        weights = weights / weights.sum()
        return np.sum(weights[:, None] * curr, axis=0)


class Policy(BasePolicy):
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

        self.model: CrossFormerModel = CrossFormerModel.load_pretrained(cfg.path, step=cfg.step)
        if self.model.dataset_statistics is None:
            raise ValueError(
                f"Checkpoint at {cfg.path} has no dataset_statistics. "
                "Re-save the checkpoint with dataset_statistics attached to the model."
            )

        heads = tuple(self.model.module.heads.keys())
        self.head_name = "single" if "single" in heads else next(iter(heads))

        self.emsembler = Ensembler(self.cfg.exp, self.cfg.chunk)
        self.horizon = 1  # 5
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.dataset_name = TASKS[cfg.task]["dataset_name"]
        self.text = TASKS[cfg.task]["text"]
        self.text = None

        from crossformer.model.components.heads.xflow import XFlowHead

        head = self.model.module.bind({"params": self.model.params}).heads[self.head_name]
        self._is_xflow = isinstance(head, XFlowHead)

        self.img_hw = {
            k: tuple(v.shape[-3:-1])
            for k, v in self.model.example_batch["observation"].items()
            if k.startswith("image_")
        }
        print(f"[debug] expected image sizes from checkpoint: {self.img_hw}")

        self.reset_history()
        if self.cfg.warmup:
            self.warmup()

    def warmup(self):
        self.task = self.model.example_batch["task"]
        warmup_batch = dict(self.model.example_batch)
        if self._is_xflow:
            head = self.model.module.bind({"params": self.model.params}).heads[self.head_name]
            warmup_batch["dof_ids"] = np.arange(head.max_dofs)
            warmup_batch["chunk_steps"] = np.arange(head.max_horizon, dtype=np.float32)
        for _ in range(self.horizon):
            print(spec(warmup_batch))
            print(self.step(warmup_batch))
        self.reset_history()

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.emsembler.reset()

    def reset(self, payload: dict):
        name = payload.get("model", "crossformer")
        if "goal" in payload:
            goal_size = self.img_hw.get("image_primary", (224, 224))
            goal_img = resize(payload["goal"]["image_primary"], goal_size)
            goal = {"image_primary": goal_img[None]}
            self.task = self.model.create_tasks(goals=goal)
        elif "text" in payload:
            text = payload["text"]
            self.text = text
            self.task = self.model.create_tasks(texts=[text])
        else:
            return {"reset": False, "error": "No goal or text provided"}

        self.reset_history()

        return {"reset": True}

    def preprocess(self, obs: dict):
        norm_stats = self.model.dataset_statistics[self.dataset_name]["proprio"]
        obs = dict(obs)
        obs["timestep_pad_mask"] = self.model.example_batch["observation"]["timestep_pad_mask"]  # dummy

        for key in obs:
            if "image" in key:
                got_hw = tuple(obs[key].shape[-3:-1])
                expect_hw = self.img_hw.get(key)
                print(f"[debug] preprocess {key}: got={got_hw} expected={expect_hw}")
                if expect_hw and got_hw != expect_hw:
                    obs[key] = resize(obs[key], expect_hw)
                    print(f"[debug] resized {key}: now={obs[key].shape[-3:-1]}")
            # normalize proprioception except for bimanual proprioception
            if "proprio" in key and key != "proprio_bimanual":
                k = key.replace("proprio_", "")
                n = norm_stats.get(k)
                if not n:
                    print(f"Warning: no normalization stats for {key}, skipping normalization")
                    continue
                obs[key] = (obs[key] - n["mean"]) / (n["std"])
        return obs

    def step(self, payload: dict):
        if payload.get("reset", False):
            return self.reset(payload)

        name = payload.get("model", "crossformer")

        # XFlowHead outputs in raw DOF space — no per-head unnormalization needed.
        # Legacy heads use per-head stats keyed by head_name.
        unnorm_stats = None
        if not self._is_xflow:
            unnorm_stats = self.model.dataset_statistics[self.dataset_name]["action"]
            unnorm_stats = drop_fn(unnorm_stats, lambda k, x: x is None or x.dtype == "O")
            unnorm_stats = jax.tree.map(lambda x: jnp.array(x), unnorm_stats)
            unnorm_stats = unnorm_stats[self.head_name]

        obs = self.preprocess(payload["observation"])

        self.history.append(obs)
        self.num_obs += 1
        obs = stack_and_pad(self.history, self.num_obs) if self.horizon > 1 else obs
        obs = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), obs)

        # XFlowHead expects dof_ids and chunk_steps from the client
        xflow_kwargs = {}
        if self._is_xflow:
            if "dof_ids" not in payload or "chunk_steps" not in payload:
                raise ValueError("XFlowHead requires 'dof_ids' and 'chunk_steps' in the payload")
            xflow_kwargs["dof_ids"] = jnp.asarray(payload["dof_ids"])[None]  # (1, A)
            xflow_kwargs["chunk_steps"] = jnp.asarray(payload["chunk_steps"])[None]  # (1, H)

        self.rng, key = jax.random.split(self.rng)
        actions = self.model.sample_actions(
            observations=obs,
            tasks=self.task,
            unnormalization_statistics=unnorm_stats,
            head_name=self.head_name,
            rng=key,
            **xflow_kwargs,
        )
        print(actions.shape)
        actions = actions[0, -1]  # one batch, last window

        actions = np.array(actions)
        print(spec({"action": actions}))
        print(actions)

        return {"actions": self.emsembler(actions[: self.cfg.chunk])}


def _wrap_preprocess_policy_v2(cfg: PolicyV2Config, core: CorePolicy, stats: dict, ds_name: str) -> BasePolicy:
    """Apply observation-side PolicyV2 wrappers to the core policy."""
    policy: BasePolicy = core

    if cfg.dtype_guard:
        policy = DtypeGuardWrapper(policy, enabled=True)

    if cfg.obs_pad:
        example_obs = jax.device_get(core.model.example_batch["observation"])
        policy = ObsPaddingWrapper(policy, example_obs)

    if cfg.resize:
        policy = ImageResizeWrapper(policy, core.img_hw)

    if cfg.proprio_norm:
        policy = ProprioNormWrapper(policy, stats[ds_name]["proprio"])

    if cfg.history > 1:
        policy = HistoryWrapper(policy, cfg.history)

    return policy


def _wrap_postprocess_policy_v2(
    cfg: PolicyV2Config, core: CorePolicy, stats: dict, ds_name: str, policy: BasePolicy
) -> BasePolicy:
    """Apply action-side PolicyV2 wrappers to the policy."""
    if cfg.denorm:
        if core.is_xflow:
            policy = XFlowDenormWrapper(policy, stats, ds_name)
        else:
            policy = LegacyDenormWrapper(policy, stats, core.head_name, ds_name)

    if cfg.ensemble:
        policy = EnsemblerWrapper(policy, cfg.exp, cfg.chunk, cfg.chunk)

    if core.is_xflow:
        policy = BodyPartGroupWrapper(policy)

    if cfg.warmup:
        core.warmup()

    return policy


def build_policy_v2(cfg: PolicyV2Config) -> BasePolicy:
    """Compose a CorePolicy with configurable wrappers from PolicyV2Config."""
    ds_name = TASKS[cfg.task]["dataset_name"]
    core = CorePolicy(cfg.path, step=cfg.step, head_name=cfg.head_name)
    stats = core.model.dataset_statistics
    policy = _wrap_preprocess_policy_v2(cfg, core, stats, ds_name)
    return _wrap_postprocess_policy_v2(cfg, core, stats, ds_name, policy)
