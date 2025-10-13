from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import time
import traceback
from typing import Any

from fastapi import FastAPI
import jax
import numpy as np
from rich.pretty import pprint
import tensorflow as tf
import tyro
import uvicorn
from webpolicy.deploy.base_policy import BasePolicy
from webpolicy.deploy.server import WebsocketPolicyServer as Server

from crossformer.cn.base import default
from crossformer.model.crossformer_model import CrossFormerModel


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
    models: str | list  # comma separated models as name : id : step
    task: str = tyro.MISSING  # task to perform

    # path to BAFL_SAVE or weights dir
    weights: str | Path = os.environ.get("BAFL_SAVE", ".")

    def __post_init__(self):
        assert self.models, "Please provide a model"
        assert self.task, "Please provide a task"
        if isinstance(self.models, str):
            self.models = [m.split(":") for m in self.models.split(",")]

        self.weights = Path(self.weights).expanduser()
        pprint(self.models)
        self.models = [(n, str(self.weights / id), s) for n, id, s in self.models]


@dataclass
class ServerConfig:
    """Server configuration"""

    policy: PolicyConfig = default(PolicyConfig())
    host: str = "0.0.0.0"  # host to run on
    port: int = 8001  # port to run on


class HttpServer:
    def __init__(self, paths, cfg: ServerCN | None = None):
        self.cfg = cfg

        self.models = {}
        for name, path, step in paths:
            self.models[name] = CrossFormerModel.load_pretrained(path, step=step)
        self.head_name = "single_arm"
        self.pred_horizon = 4
        self.exp_weight = 0
        self.horizon = 1  # 5
        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.tasks = {
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
        self.dataset_name = self.tasks[cfg.task]["dataset_name"]
        self.text = self.tasks[cfg.task]["text"]

        self.reset_history()

        # trigger compilation
        for name in self.models:
            payload = {
                "text": self.text,
                "model": name,
            }
            self.reset(payload)
            payload = {
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
                "dataset_name": self.dataset_name,
            }

            for _ in range(self.horizon):
                start = time.time()
                print(self.sample_actions(payload))
                print(time.time() - start)

        self.reset_history()

    def run(self, port=8001, host="0.0.0.0"):
        self.app = FastAPI()
        self.app.post("/query")(self.sample_actions)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.pred_horizon)

    def reset(self, payload: dict[Any, Any]):
        model_name = payload.get("model", "crossformer")
        if "goal" in payload:
            goal_img = resize(payload["goal"]["image_primary"])
            goal = {"image_primary": goal_img[None]}
            self.task = self.models[model_name].create_tasks(goals=goal)
        elif "text" in payload:
            text = payload["text"]
            self.text = text
            self.task = self.models[model_name].create_tasks(texts=[text])
        else:
            raise ValueError

        self.reset_history()

        return "reset"

    def sample_actions(self, payload: dict[Any, Any]):
        try:
            model_name = payload.get("model", "crossformer")

            obs = payload["observation"]
            for key in obs:
                if "image" in key:
                    obs[key] = resize(obs[key])
                # NOTE... single proprio might fail if not processed accordingly
                # normalize proprioception expect for bimanual proprioception
                if "proprio" in key and key != "proprio_bimanual":
                    proprio_normalization_statistics = self.models[model_name].dataset_statistics[self.dataset_name][
                        key
                    ]
                    obs[key] = (obs[key] - proprio_normalization_statistics["mean"]) / (
                        proprio_normalization_statistics["std"]
                    )

            self.history.append(obs)
            self.num_obs += 1
            obs = stack_and_pad(self.history, self.num_obs)

            # add batch dim
            obs = jax.tree.map(lambda x: x[None], obs)

            unnormalization_statistics = self.models[model_name].dataset_statistics[self.dataset_name]["action"]

            self.rng, key = jax.random.split(self.rng)
            actions = self.models[model_name].sample_actions(
                obs,
                self.task,
                unnormalization_statistics,
                head_name=self.head_name,
                rng=key,
            )
            print(actions.shape)
            actions = actions[0, -1]  # one batch, last window

            actions = np.array(actions)
            print(f"actions: {actions.shape}")

            # whether to temporally ensemble the action predictions or return the full chunk
            if not payload.get("ensemble", True):
                return json_response(actions)

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

            return json_response(action)
        except:
            print(traceback.format_exc())
            return "error"


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
            payload = {
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
                "dataset_name": self.dataset_name,
            }

            for _ in range(self.horizon):
                print(self.sample_actions(payload))

        self.reset_history()

    def reset_history(self):
        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0
        self.act_history = deque(maxlen=self.pred_horizon)

    def reset(self, payload: dict | None = None):
        pass

    def infer(self, obs: dict):
        pass


def _main(cfg: ServerCN):
    tf.config.set_visible_devices([], "GPU")

    # DEPRICATE name, path, step
    # root = osp.expanduser("~/data/bafl") # for carina
    # paths = [
    # ("crossformer", "hf://rail-berkeley/crossformer", None),
    # v3
    # ("bafl", osp.join(root, "experiment_20241203_193649"), 14_000), # v3-lift
    # ("bafl", osp.join(root, "experiment_20241204_141047"), 150_000), # v3-lift
    # ("bafl", osp.join(root, "experiment_20241211_173647"), 290_000),  # v3-lift
    # ]

    server = HttpServer(cfg.models, cfg=cfg)
    server.run(cfg.port, cfg.host)


def main(cfg: Config):
    pprint(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.policy.device)

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
    main(tyro.cli(Config))


if __name__ == "__main__":
    main(tyro.cli(ServerCN))
