from collections import deque, OrderedDict
import os
import os.path as osp
from typing import Optional, Sequence

from absl import app, flags, logging
import gymnasium as gym
from improve.env.action_rescale import ActionRescaler
from improve.fm.batch_octo import BatchedActionEnsembler
import improve.wrapper as W
import jax
import matplotlib.pyplot as plt
import numpy as np
from octo.model.octo_model import OctoModel
from octo.utils import gym_wrappers as GW
import simpler_env as simpler
from simpler_env.utils.action.action_ensemble import ActionEnsembler

# from improve.fm.batch_octo import BatchedOctoInference
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecVideoRecorder,
)
import tensorflow as tf
from tqdm import tqdm
from transformers import AutoTokenizer
from transforms3d.euler import euler2axangle

tf.config.experimental.set_visible_devices([], "GPU")

class BatchedActionEnsembler(ActionEnsembler):
    def __init__(self, pred_action_horizon, action_ensemble_temp=0.0, batch_size=1):
        self.ensemblers = [
            ActionEnsembler(pred_action_horizon, action_ensemble_temp)
            for _ in range(batch_size)
        ]

    def reset(self):
        for ensembler in self.ensemblers:
            ensembler.reset()

    def ensemble_action(self, cur_action):
        return np.stack(
            [
                ensembler.ensemble_action(cur_action[i])
                for i, ensembler in enumerate(self.ensemblers)
            ]
        )


class PolicyStepper:

    def __init__(self, model_type, dataset_id, func=None, transform=None, task=None):
        self.model_type = model_type
        self.dataset_id = dataset_id

        if self.model_type == "func":
            self.init_data_stats()
            self.func = func
            self.transform = transform
            # transform can do this for now
            # assert task is not None
            # self._task = task  # task embedding vector

        else:
            self.init_model()

    def init_model(self):

        if self.model_type in ["octo-base", "octo-small"]:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{self.model_type}"
            self.tokenizer, self.tokenizer_kwargs = None, None
            self.model = OctoModel.load_pretrained(self.model_type)
            self.action_mean = self.model.dataset_statistics[self.dataset_id]["action"][
                "mean"
            ]
            self.action_std = self.model.dataset_statistics[self.dataset_id]["action"][
                "std"
            ]
            self.automatic_task_creation = True

        else:
            # custom model path
            self.model_type = self.model_type
            self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.tokenizer_kwargs = {
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            }
            self.model = tf.saved_model.load(self.model_type)
            self.automatic_task_creation = False

            self.init_data_stats()

    def init_data_stats(self):
        if self.dataset_id == "bridge_dataset":
            self.action_mean = np.array(
                [
                    0.00021161,
                    0.00012614,
                    -0.00017022,
                    -0.00015062,
                    -0.00023831,
                    0.00025646,
                    0.0,
                ]
            )
            self.action_std = np.array(
                [
                    0.00963721,
                    0.0135066,
                    0.01251861,
                    0.02806791,
                    0.03016905,
                    0.07632624,
                    1.0,
                ]
            )

        elif self.dataset_id == "fractal20220817_data":
            self.action_mean = np.array(
                [
                    0.00696389,
                    0.00627008,
                    -0.01263256,
                    0.04330839,
                    -0.00570499,
                    0.00089247,
                    0.0,
                ]
            )
            self.action_std = np.array(
                [
                    0.06925472,
                    0.06019009,
                    0.07354742,
                    0.15605888,
                    0.1316399,
                    0.14593437,
                    1.0,
                ]
            )

        else:
            msg = f"{self.dataset_id} not supported yet for custom octo model checkpoints."
            raise NotImplementedError(msg)

    @property
    def task(self):
        if self.model_type == "func":
            return self._task

        if self.automatic_task_creation:
            self._task = self.model.create_tasks(texts=[task_description])
        else:
            self._task = self.tokenizer(task_description, **self.tokenizer_kwargs)
        return self._task

    def fwd(self, batch, rng, key):
        """only for DDP speedup"""

        if self.automatic_task_creation:
            norm_raw_actions = self.model.sample_actions(
                batch["observation"],
                batch["task"],
                rng=key,
            )

        else:
            input_observation = {"image_primary": images, "timestep_pad_mask": pad_mask}
            input_observation = {
                "observations": input_observation,
                "tasks": {"language_instruction": self._task},
                "rng": np.concatenate([rng, key]),
            }
            norm_raw_actions = self.model.lc_ws2(input_observation)[:, :, :7]

        return norm_raw_actions

    def __call__(self, batch, rng, key):
        """steps with either model or step function and returns result"""

        if self.model_type == "func":
            batch = self.transform(batch) if self.transform is not None else batch
            return self.func(batch)

        else:
            return self.fwd(batch, rng, key)


class OXESimplerInference:
    def __init__(
        self,
        stepper: PolicyStepper,
        # model_type: str = "octo-base",
        policy_setup: str = "widowx_bridge",
        horizon: int = 2,
        pred_action_horizon: int = 4,
        exec_horizon: int = 1,
        image_size: int = 256,
        action_scale: float = 1.0,
        init_rng: int = 0,
        batch_size: int = 8,
    ) -> None:

        self.policy_setup = policy_setup
        self.init_policy()

        self.stepper = stepper
        self.action_mean = self.stepper.action_mean
        self.action_std = self.stepper.action_std

        self.batch_size = batch_size

        # init observations
        self.image_size = image_size
        self.horizon = horizon

        # self.task = None
        # self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.num_image_history = 0

        # init actions
        self.action_scale = action_scale
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.rng = jax.random.PRNGKey(init_rng)
        for _ in range(5):
            # the purpose of this for loop is just to match octo server's inference seeds
            self.rng, _key = jax.random.split(self.rng)  # each shape [2,]

        # init sticky gripper (only used if num_repeat > 1)?
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

        if self.action_ensemble:
            self.action_ensembler = BatchedActionEnsembler(
                self.pred_action_horizon,
                self.action_ensemble_temp,
                batch_size=self.batch_size,
            )
        else:
            self.action_ensembler = None

        # needed for now. could be problem later
        tf.config.experimental.set_visible_devices([], "GPU")

    def init_policy(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.policy_setup == "widowx_bridge":
            self.dataset_id = "bridge_dataset"
            self.action_ensemble = True
            self.action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1

        elif self.policy_setup == "google_robot":
            self.dataset_id = "fractal20220817_data"
            self.action_ensemble = True
            self.action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15

        else:
            msg = f"Policy setup {self.policy_setup} not supported for octo models."
            raise NotImplementedError(msg)

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        ax = 1  # 0 if self.batch_size == 1 else 1
        images = np.stack(self.image_history, axis=ax)
        horizon = len(self.image_history)
        # note: this should be of float type, not a bool type
        pad_mask = np.ones(horizon, dtype=np.float64)
        pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # pad_mask = np.ones(self.horizon, dtype=np.float64) # note: this should be of float type, not a bool type
        # pad_mask[:self.horizon - self.num_image_history] = 0
        return images, pad_mask

    def reset(self, task_description: str) -> None:
        # self.task = self.stepper.task

        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        # self.gripper_is_closed = False
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, descs: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (B, H, W, 3), uint8
            descs: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if descs is not None and descs != self.descs:
            self.reset(descs)  # task description has changed; reset the policy state

        # then this is actually batch or obs not image
        if isinstance(image, (dict, OrderedDict)):
            try:
                image = image["observation"]["image_primary"]
            except:
                image = image["simpler-img"]

        assert image.dtype == np.uint8, image.dtype
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, pad_mask = self._obtain_image_history_and_mask()

        pad_mask = np.repeat(pad_mask[None, :], self.batch_size, axis=0)

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        # print("octo local rng", self.rng, key)

        batch = {"observation": {"image_primary": images, "pad_mask": pad_mask}}
        norm_raw_actions = self.stepper(batch, rng=self.rng, key=key)

        assert norm_raw_actions.shape == (self.batch_size, self.pred_action_horizon, 7)

        if self.action_ensemble:
            norm_raw_actions = self.action_ensembler.ensemble_action(norm_raw_actions)

        raw_actions = norm_raw_actions * self.action_std[None] + self.action_mean[None]
        raw_action = {
            "world_vector": np.array(raw_actions[:, :3]),
            "rotation_delta": np.array(raw_actions[:, 3:6]),
            "open_gripper": np.array(
                raw_actions[:, 6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )

        axangles = []
        for rotation_delta in action_rotation_delta:
            roll, pitch, yaw = rotation_delta
            ax, angle = euler2axangle(roll, pitch, yaw)
            axangle = ax * angle
            axangles.append(axangle[None])

        action["rot_axangle"] = np.concatenate(axangles, axis=0) * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # see simpler.OctoInference for alternative implementation option
            # wow. what a mess when vectorized
            # alternative implementation
            relative_gripper_action = np.where(
                np.isnan(self.previous_gripper_action),
                np.zeros_like(self.previous_gripper_action),
                self.previous_gripper_action - current_gripper_action,
            )
            self.previous_gripper_action = current_gripper_action

            to_stick = np.logical_and(
                np.abs(relative_gripper_action) > 0.5,
                (self.sticky_action_is_on is False),
            )
            self.sticky_action_is_on = np.where(
                to_stick, True, self.sticky_action_is_on
            )
            self.sticky_gripper_action = np.where(
                to_stick, relative_gripper_action, self.sticky_gripper_action
            )

            self.gripper_action_repeat += self.sticky_action_is_on.astype(int)
            relative_gripper_action = np.where(
                self.sticky_action_is_on,
                self.sticky_gripper_action,
                relative_gripper_action,
            )

            reset = self.gripper_action_repeat == self.sticky_gripper_num_repeat
            self.sticky_action_is_on = np.where(reset, False, self.sticky_action_is_on)
            self.gripper_action_repeat = np.where(reset, 0, self.gripper_action_repeat)
            self.sticky_gripper_action = np.where(
                reset, 0.0, self.sticky_gripper_action
            )

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            # binarize gripper action to 1 (open) and -1 (close)
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0] * self.batch_size)

        return raw_action, action

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)


class EvalCallback:

    def __init__(self, venv, step_func, transform=None, oxes=None):
        self.venv = venv
        self.step_func = step_func
        self.transform = transform
        self.oxes = oxes
        self.rescaler = ActionRescaler()

    def __call__(self, i=None):

        if self.oxes is not None:
            self.oxes.reset()

        size = self.venv.num_envs
        dones = np.zeros(size, dtype="int")

        rewards = np.zeros(size)
        lengths = np.zeros(size, dtype="int")
        obs = self.venv.reset()  # venv reset has no info

        obs = self.transform(obs) if self.transform is not None else obs
        obs = self.oxes.pre_step(obs) if self.oxes is not None else obs

        states = None
        episode_starts = np.ones((self.venv.num_envs,), dtype=bool)

        # print(jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), obs))

        bar = tqdm(range(120), desc="eval callback", total=120, leave=False)
        for _ in bar:
            if dones.all():
                break

            # actions = np.array(self.step_func(obs))
            actions = self.step_func(obs)
            if self.oxes is not None:
                raw, actions = self.oxes(actions)

            actions = self.rescaler.dict2act(actions)

            # names = ["x", "y", "z", "yaw", "pitch", "roll", "gripper"]
            # for i, n in enumerate(names):
            # wandb.log({f"pred/{n}": wandb.Histogram(actions[:, i])})

            obs, rew, done, info = self.venv.step(actions)
            obs = self.transform(obs) if self.transform is not None else obs
            obs = self.oxes.pre_step(obs) if self.oxes is not None else obs
            # done = terminated or truncated

            rewards = rewards + (rew * np.logical_not(rewards))
            lengths = lengths + (1 * np.logical_not(rewards))
            dones = np.logical_or(dones, done)

            bar.set_description(f"rewards: {rewards.sum()}")
            bar.update()

        stats = {"SR": rewards.mean(), "length": lengths.mean()}
        print(stats)
        return stats


def step_randoms(*args):
    import numpy as np

    return np.random.rand(size, 4, 7)


from typing import Tuple, Union


def isimg(o):
    if isinstance(o, np.ndarray) and o.ndim == 3:
        return o.shape[-1] in [1, 3, 4]
    return False


class ResizeImageWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Union[Tuple, Sequence[Tuple]]],
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        spaces = self.observation_space.spaces
        self.resize_size = resize_size

        logging.info(f"Resizing images:")
        for k, v in self.observation_space.sample().items():
            if isimg(v):
                spaces[k] = gym.spaces.Box(
                    low=0,
                    high=255,  # pixel brightness
                    shape=resize_size + (3,),
                    dtype=np.uint8,
                )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        for k, v in observation.items():
            if isimg(v):
                image = tf.image.resize(
                    v, size=self.resize_size, method="lanczos3", antialias=True
                )
                image = tf.cast(
                    tf.clip_by_value(tf.round(image), 0, 255), tf.uint8
                ).numpy()
                observation[k] = image
        return observation


def mk_envs(tasks, n_envs=16, image_size=224):
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

    import random

    def _init() -> gym.Env:

        t = random.choice(tasks)
        print(f'Creating env for task "{t}"')
        env = simpler.make(
            t,
            # cant find simpler-img if you specify the mode
            render_mode="cameras",
            # max_episode_steps=max_episode_steps,
            renderer_kwargs={
                "offscreen_only": True,
                "device": "cuda:0",
            },
            # **extra,
        )

        # oxes does this
        # env = W.ActionRescaleWrapper(env)
        # env = W.StickyGripperWrapper(env, task=cfg.task)

        env = W.ExtraObservationWrapper(env, use_image=True)
        env = W.FlattenKeysWrapper(env)
        env = W.FilterKeysWrapper(env, keys=["simpler-img"])
        env = W.SuccessInfoWrapper(env)
        # print(env.observation_space)

        # TODO octo needs gymnasium as gym

        # requires key.startswith('image_')
        env = ResizeImageWrapper(env, (image_size, image_size))
        # env = ResizeImageWrapper(env, (256, 256))

        # env = GW.HistoryWrapper(env, 2)
        # env = GW.TemporalEnsembleWrapper(env, 4)

        # env = GW.RHCWrapper(env, 2)
        return env

    # venv = SubprocVecEnv([lambda: _init(t) for t in iter(tasks)])
    venv = SubprocVecEnv([_init for _ in range(n_envs)])
    # venv = VecMonitor(venv)  # attach this so SB3 can log reward metrics

    venv.seed(0)
    venv.reset()
    venv = W.VecRecord(venv, osp.join("log_dir", "train"), use_wandb=True)
    return venv


def main():
    import simpler_env as simpler

    tasks = [e for e in simpler.ENVIRONMENTS if "widowx" in e]
    # replicates a few times
    tasks = tasks 
    print(tasks)
    print(len(tasks))
    venv = mk_envs(tasks, 16)
    instructions = venv.env_method("get_language_instruction")
    from pprint import pprint
    pprint(instructions)

if __name__ == "__main__":
    main()
