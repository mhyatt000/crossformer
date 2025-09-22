from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path

import jax
import numpy as np
from rich.pretty import pprint
import tensorflow as tf
import tyro
from webpolicy.deploy.base_policy import BasePolicy
from webpolicy.deploy.server import WebsocketPolicyServer as Server

from crossformer.cn.base import default, CN
from crossformer.model.crossformer_model import CrossFormerModel

tf.config.set_visible_devices([], "GPU")


def resize(img, size=(224, 224)):
    img = tf.image.resize(img, size=size, method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8).numpy()


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


@dataclass
class PolicyConfig:
    models: str | list  = "" # comma separated models as name : id : step
    task: str | None = None # task to perform

    # path to BAFL_SAVE or weights dir
    weights: str | Path = os.environ.get("BAFL_SAVE", ".")

    def __post_init__(self):
        if self.models and isinstance(self.models, str):
            self.models = [m.split(":") for m in self.models.split(",")]

        self.weights = Path(self.weights).expanduser()
        self.models = [(n, str(self.weights / id), s) for n, id, s in self.models]

    def verify(self):
        assert self.models, "Please provide a model"
        assert self.task, "Please provide a task"
        for name, path, step in self.models:
            assert Path(path).exists(), f"Model path {path} does not exist"


@dataclass
class ServerConfig(CN):
    """Server configuration"""

    policy: PolicyConfig = default(PolicyConfig())
    host: str = "0.0.0.0"  # host to run on
    port: int = 8001  # port to run on


def make_exbatch(name, dataset_name):
    exbatch = {
        "observation": {
            "proprio_bimanual": np.zeros((14,)),
            "proprio_single": np.zeros((6,)),
            "image_primary": np.zeros((224, 224, 3)),
            "image_high": np.zeros((224, 224, 3)),
            "image_side": np.zeros((224, 224, 3)),
            "image_left_wrist": np.zeros((224, 224, 3)),
            # "image_right_wrist": np.zeros((224, 224, 3)),
        },
        "modality": "l",  # l for language or v for vision
        "ensemble": True,
        "model": name,
        "dataset_name": dataset_name,
    }
    return exbatch


TASKS = {
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

        self.models = {}
        for name, path, step in cfg.models:
            self.models[name] = CrossFormerModel.load_pretrained(path, step=step)
        self.head_name = "single_arm"
        self.pred_horizon = 4
        self.exp_weight = 0
        self.horizon = 1  # 5
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.dataset_name = TASKS[cfg.task]["dataset_name"]
        self.text = TASKS[cfg.task]["text"]

        self.reset_history()

        # trigger compilation
        for name in self.models:
            payload = {
                "text": self.text,
                "model": name,
            }
            self.reset(payload)

            for _ in range(self.horizon):
                print(self.infer(make_exbatch(name, self.dataset_name)))

        self.reset_history()

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.pred_horizon)

    def reset(self, payload: dict):
        name = payload.get("model", "crossformer")
        if "goal" in payload:
            goal_img = resize(payload["goal"]["image_primary"])
            goal = {"image_primary": goal_img[None]}
            self.task = self.models[name].create_tasks(goals=goal)
        elif "text" in payload:
            text = payload["text"]
            self.text = text
            self.task = self.models[name].create_tasks(texts=[text])
        else:
            return {"reset": False, "error": "No goal or text provided"}

        self.reset_history()

        return {"reset": True}

    def infer(self, payload: dict):
        name = payload.get("model", "crossformer")

        norm_stats = self.models[name].dataset_statistics[self.dataset_name]
        unnorm_stats = self.models[name].dataset_statistics[self.dataset_name]

        obs = payload["observation"]
        for key in obs:
            if "image" in key:
                obs[key] = resize(obs[key])
            # NOTE... single proprio might fail if not processed accordingly
            # normalize proprioception expect for bimanual proprioception
            if "proprio" in key and key != "proprio_bimanual":
                obs[key] = (obs[key] - norm_stats[key]["mean"]) / (
                    norm_stats[key]["std"]
                )

        self.history.append(obs)
        self.num_obs += 1
        obs = stack_and_pad(self.history, self.num_obs)

        obs = jax.tree.map(lambda x: x[None], obs)  # add batch dim

        self.rng, key = jax.random.split(self.rng)
        actions = self.models[name].sample_actions(
            obs,
            self.task,
            unnorm_stats["action"],
            head_name=self.head_name,
            rng=key,
        )
        print(actions.shape)
        actions = actions[0, -1]  # one batch, last window

        actions = np.array(actions)
        print(f"actions: {actions.shape}")

        # whether to temporally ensemble the action predictions or return the full chunk
        if not payload.get("ensemble", True):
            return {"action": actions}

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)
        print(f"num_actions: {num_actions}")

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
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
