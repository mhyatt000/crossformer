from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rich.pretty import pprint
import tensorflow as tf
import tyro
from webpolicy.deploy.base_policy import BasePolicy
from webpolicy.deploy.server import WebsocketPolicyServer as Server

from crossformer.cn.base import CN, default
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils import spec

tf.config.set_visible_devices([], "GPU")


def resize(img, size=(224, 224)):
    if stack := len(img.shape) == 5:  # flatten unflatten
        n, m, h, w, c = img.shape
        img = img.reshape(n * m, h, w, c)

    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    h, w = size  # no longer the initial h,w
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()
    if stack:
        img = img.reshape(n, m, h, w, c)
    return img


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
    path: str | None = None
    task: str | None = None  # task to perform
    step: int | None = None  # step to load

    def verify(self):
        # assert self.models, "Please provide a model"
        assert self.task, "Please provide a task"
        assert self.path and Path(self.path).resolve().expanduser().exists(), f"Model path {self.path} does not exist"

    """
    # path to BAFL_SAVE or weights dir
    weights: str | Path = os.environ.get("BAFL_SAVE", ".")

    def __post_init__(self):
        if self.models and isinstance(self.models, str):
            self.models = [m.split(":") for m in self.models.split(",")]

        self.weights = Path(self.weights).expanduser()
        self.models = [(n, str(self.weights / id), s) for n, id, s in self.models]
    """


@dataclass
class ServerConfig(CN):
    """Server configuration"""

    policy: PolicyConfig = default(PolicyConfig())
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


class Policy(BasePolicy):
    def __init__(self, cfg: PolicyConfig):
        self.cfg = cfg

        self.model: CrossFormerModel = CrossFormerModel.load_pretrained(cfg.path, step=cfg.step)
        self.head_name = "single_arm"
        self.pred_horizon = 4
        self.exp_weight = 0
        self.horizon = 1  # 5
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.dataset_name = TASKS[cfg.task]["dataset_name"]
        self.text = TASKS[cfg.task]["text"]
        self.exob = self.model.example_batch["observation"]

        self.reset_history()
        self.reset({"text": self.text})  # trigger compilation

        # for _ in range(self.horizon):
        pprint(spec(self.model.example_batch))
        print(self.infer(self.model.example_batch))

        self.reset_history()

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.pred_horizon)

    def reset(self, payload: dict):
        name = payload.get("model", "crossformer")
        if "goal" in payload:
            imsize = self.exob["image_primary"].shape[-2]
            goal_img = resize(payload["goal"]["image_primary"], size=(imsize, imsize))
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

    def infer(self, payload: dict):
        if payload.get("reset", False):
            return self.reset(payload)

        name = payload.get("model", "crossformer")

        norm_stats = self.model.dataset_statistics[self.dataset_name]
        unnorm_stats = self.model.dataset_statistics[self.dataset_name]

        obs = payload["observation"]
        obs["timestep_pad_mask"] = self.model.example_batch["observation"]["timestep_pad_mask"]  # dummy
        imsize = self.exob["image_primary"].shape[-2]
        for key in obs:
            if "image" in key:
                obs[key] = resize(obs[key], size=(imsize, imsize))
            # NOTE... single proprio might fail if not processed accordingly
            # normalize proprioception expect for bimanual proprioception
            if "proprio" in key and key != "proprio_bimanual":
                obs[key] = (obs[key] - norm_stats[key]["mean"]) / (norm_stats[key]["std"])

        pprint(spec(obs))

        self.history.append(obs)
        self.num_obs += 1
        obs = stack_and_pad(self.history, self.num_obs) if self.horizon > 1 else obs
        obs = jax.tree_map(lambda x: jax.device_put(jnp.asarray(x)), obs)

        self.rng, key = jax.random.split(self.rng)
        actions = self.model.sample_actions(
            observations=obs,
            tasks=self.task,
            unnormalization_statistics=unnorm_stats["action"],
            head_name=self.head_name,
            rng=key,
        )
        pprint(actions.shape)
        actions = actions[0, -1]  # one batch, last window

        actions = np.array(actions)
        pprint(spec({"action": actions}))
        # pprint(actions)

        # whether to temporally ensemble the action predictions or return the full chunk
        if not payload.get("ensemble", True):
            return {"action": actions}

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)
        print(f"num_actions: {num_actions}")

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.act_history)]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return {"action": action}


def main(cfg: ServerConfig):
    pprint(cfg)
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
