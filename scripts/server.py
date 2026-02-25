from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rich import print
import tyro
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.spec import spec
from crossformer.utils.tree.core import drop_fn


def resize(img, size=(224, 224)):
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
        assert self.path and Path(self.path).resolve().expanduser().exists(), f"Model path {self.path} does not exist"
        assert self.task in TASKS, f"Unknown task '{self.task}'. Choose from: {sorted(TASKS)}"


@dataclass
class ServerConfig:
    """Server configuration"""

    policy: PolicyConfig
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
        self.head_name = "single_arm"

        self.emsembler = Ensembler(self.cfg.exp, self.cfg.chunk)
        self.horizon = 1  # 5
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.dataset_name = TASKS[cfg.task]["dataset_name"]
        self.text = TASKS[cfg.task]["text"]
        self.text = None

        self.reset_history()
        if self.cfg.warmup:
            self.warmup()

    def warmup(self):
        # self.reset({"text": self.text})  # trigger compilation
        self.reset(self.model.example_batch)  # trigger compilation
        for _ in range(self.horizon):
            print(spec(self.model.example_batch))
            print(self.step(self.model.example_batch))
        self.reset_history()

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.emsembler.reset()

    def reset(self, payload: dict):
        name = payload.get("model", "crossformer")
        if "goal" in payload:
            goal_img = resize(payload["goal"]["image_primary"])
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
        print(norm_stats.keys())
        obs = dict(obs)
        obs["timestep_pad_mask"] = self.model.example_batch["observation"]["timestep_pad_mask"]  # dummy
        for key in obs:
            if "image" in key:
                obs[key] = resize(obs[key])
            # NOTE... single proprio might fail if not processed accordingly
            # normalize proprioception expect for bimanual proprioception
            if "proprio" in key and key != "proprio_bimanual":
                k = key.replace("proprio_", "")  # TODO @agent fix for clarity, this is a bit hacky
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

        unnorm_stats = self.model.dataset_statistics[self.dataset_name]["action"]
        obs = self.preprocess(payload["observation"])

        self.history.append(obs)
        self.num_obs += 1
        obs = stack_and_pad(self.history, self.num_obs) if self.horizon > 1 else obs
        obs = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), obs)

        tspec = lambda a: jax.tree_map(lambda x: type(x), a)
        unnorm_stats = drop_fn(unnorm_stats, lambda x: x.dtype == "O")
        unnorm_stats = jax.tree.map(lambda x: jnp.array(x), unnorm_stats)
        print(tspec(self.task))
        self.task = jnp.zeros((1, 512))
        # print(tspec(obs))
        # quit()

        jax.config.update("jax_disable_jit", True)
        self.rng, key = jax.random.split(self.rng)
        # sample = pack_np2jax(self.model.sample_actions)
        actions = self.model.sample_actions(
            observations=obs,
            tasks=self.task,
            # unnormalization_statistics=unnorm_stats,
            head_name=self.head_name,
            rng=key,
        )
        print(actions.shape)
        actions = actions[0, -1]  # one batch, last window

        actions = np.array(actions)
        print(spec({"action": actions}))
        print(actions)

        return {"actions": self.emsembler(actions[: self.cfg.chunk])}


def main(cfg: ServerConfig):
    print(cfg)
    cfg.policy.verify()

    policy = Policy(cfg.policy)
    server = Server(
        policy,
        host=cfg.host,
        port=cfg.port,
        metadata=None,
    )
    print("serving on", cfg.host, cfg.port)
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
