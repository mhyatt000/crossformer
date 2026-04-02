from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from webpolicy.base_policy import BasePolicy

from crossformer.embody import DOF
from crossformer.model.components.heads.dof import pad_chunk_steps, pad_dof_ids
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run._wrappers import _resize
from crossformer.run.train_step import lookup_guide
from crossformer.run.wrappers import PolicyWrapper
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer


class ModelPolicy(BasePolicy):
    """Inference policy that loads a checkpoint and runs the xflow forward pass.

    Expects preprocessed batches with observation, task, and timestep_pad_mask.
    DOF ids (j0..j6 + gripper) and chunk steps (arange 20) are hardcoded.
    """

    # j0..j6 + gripper — the 8 DOFs used for arm+gripper inference
    _DEFAULT_DOF_IDS: tuple[int, ...] = (
        DOF["j0"],
        DOF["j1"],
        DOF["j2"],
        DOF["j3"],
        DOF["j4"],
        DOF["j5"],
        DOF["j6"],
        DOF["gripper"],
    )
    _DEFAULT_CHUNK_STEPS: tuple[float, ...] = tuple(float(i) for i in range(20))

    def __init__(
        self,
        path: str,
        *,
        step: int | None = None,
        head_name: str = "xflow",
        guide_keys: tuple[str, ...] = ("action.position", "action.orientation"),
        use_guidance: bool = True,
    ):
        self.model: CrossFormerModel = CrossFormerModel.load_pretrained(path, step=step)
        self.params = self.model.params
        self.head_name = head_name
        self.guide_keys = guide_keys
        self.use_guidance = use_guidance
        self.rng = jax.random.PRNGKey(0)

        module = self.model.module
        head = module.bind({"params": self.params}).heads[head_name]
        self._dof_ids_1 = jnp.asarray(pad_dof_ids(self._DEFAULT_DOF_IDS, head.max_dofs))[None]  # (1, max_dofs)
        self._chunk_steps_1 = jnp.asarray(
            pad_chunk_steps(self._DEFAULT_CHUNK_STEPS, head.max_horizon), dtype=jnp.float32
        )[None]  # (1, max_horizon)

        @partial(jax.jit, static_argnames=("accumulate",))
        def _jit_step(params, obs, task, timestep_pad_mask, dof_ids, chunk_steps, guide_input, rng, accumulate=False):
            bound = module.bind({"params": params})
            transformer_outputs = bound.crossformer_transformer(obs, task, timestep_pad_mask, train=False)
            pred = bound.heads[head_name].predict_action(
                transformer_outputs,
                rng=rng,
                dof_ids=dof_ids,
                chunk_steps=chunk_steps,
                train=False,
                guide_input=guide_input,
                accumulate=accumulate,
            )
            return pred, dof_ids

        self._jit_step = _jit_step

    def warmup(self, accumulate: bool = True) -> None:
        """Trigger JIT compilation for the given accumulate mode using example_batch."""
        self.step(self.model.example_batch, accumulate=accumulate)

    def reset(self, payload: dict | None = None) -> dict | None:
        return None

    def step(self, payload: dict, *, accumulate: bool = False) -> dict:
        self.rng, key = jax.random.split(self.rng)
        obs = payload["observation"]
        task = payload.get("task", {"pad_mask_dict": {}})
        B = jnp.asarray(obs["timestep_pad_mask"]).shape[0]
        dof_ids = jnp.tile(self._dof_ids_1, (B, 1))
        chunk_steps = jnp.tile(self._chunk_steps_1, (B, 1))
        guide_input = lookup_guide(payload, self.guide_keys) if self.use_guidance else None
        pred, dof_ids = self._jit_step(
            self.params,
            obs,
            task,
            obs["timestep_pad_mask"],
            dof_ids,
            chunk_steps,
            guide_input,
            key,
            accumulate=accumulate,
        )
        return {"actions": jax.device_get(pred), "dof_ids": jax.device_get(dof_ids)}


class ActionDenormWrapper(PolicyWrapper):
    """Wraps a policy and denormalizes the 'actions' field in the result.

    Handles step shapes (B, W, H, A) and flow shapes (T, B, W, H, A) — B is
    at axis 0 for step and axis 1 for flow (ndim==5).  Denorm is applied
    per-sample using dof_ids and dataset_name (fixed at construction time).
    """

    def __init__(self, inner, stats, dataset_name: str):
        self.inner = inner
        self._denorm = ActionBatchDenormalizer(stats)
        self.dataset_name = dataset_name

    def step(self, payload: dict, **kwargs) -> dict:
        result = dict(self.inner.step(payload, **kwargs))
        B = np.asarray(result["dof_ids"]).shape[0]
        ds_names = [self.dataset_name] * B
        result["actions"] = self._denorm_actions(result["actions"], result["dof_ids"], ds_names)
        return result

    def _denorm_actions(self, actions, dof_ids, ds_names: list[str]) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.float32)
        ids = np.asarray(dof_ids)  # (B, A)
        flow = arr.ndim == 5  # (T, B, W, H, A) vs (B, *, A)
        if flow:
            arr = np.moveaxis(arr, 1, 0)  # → (B, T, W, H, A)
        orig_shape = arr.shape
        flat = arr.reshape(len(ds_names), -1, arr.shape[-1])  # (B, N, A)
        out = flat.copy()
        for b, ds_name in enumerate(ds_names):
            for k in range(flat.shape[1]):
                out[b, k] = self._denorm.denormalize_slot(flat[b, k], ids[b], ds_name)
        arr = out.reshape(orig_shape)
        return np.moveaxis(arr, 0, 1) if flow else arr


class CorePolicy(BasePolicy):
    """Minimal inference policy — load model, create tasks, sample actions.

    All observation preprocessing and action postprocessing is handled by
    wrapping this with ``PolicyWrapper`` subclasses from ``run.wrappers``.
    """

    def __init__(self, path: str, step: int | None = None, head_name: str | None = None):
        self.model: CrossFormerModel = CrossFormerModel.load_pretrained(path, step=step)
        if self.model.dataset_statistics is None:
            raise ValueError(
                f"Checkpoint at {path} has no dataset_statistics. "
                "Re-save the checkpoint with dataset_statistics attached."
            )

        heads = tuple(self.model.module.heads.keys())
        if head_name is None:
            head_name = "single" if "single" in heads else next(iter(heads))
        self.head_name = head_name

        self.task = None
        self.rng = jax.random.PRNGKey(0)

        self.img_hw: dict[str, tuple[int, int]] = {
            k: tuple(v.shape[-3:-1])
            for k, v in self.model.example_batch["observation"].items()
            if k.startswith("image_")
        }

    @property
    def is_xflow(self) -> bool:
        from crossformer.model.components.heads.xflow import XFlowHead

        head = self.model.module.bind({"params": self.model.params}).heads[self.head_name]
        return isinstance(head, XFlowHead)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, payload: dict | None = None) -> dict:
        if payload is None:
            # issue/59: fall back to checkpoint default, not empty dict
            self.task = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), self.model.example_batch["task"])
            return {"reset": True}

        if "goal" in payload:
            size = self.img_hw.get("image_primary", (224, 224))
            goal_img = _resize(payload["goal"]["image_primary"], size)
            self.task = self.model.create_tasks(goals={"image_primary": goal_img[None]})
        elif "text" in payload:
            self.task = self.model.create_tasks(texts=[payload["text"]])
        else:
            return {"reset": False, "error": "No goal or text provided"}

        return {"reset": True}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, payload: dict) -> dict:
        if payload.get("reset", False):
            return self.reset(payload)

        if "task" in payload:
            self.task = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), payload["task"])

        obs = payload.get("observation", payload)
        obs = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), obs)

        kwargs = {}
        if self.is_xflow:
            if "dof_ids" not in payload or "chunk_steps" not in payload:
                raise ValueError("XFlowHead requires 'dof_ids' and 'chunk_steps' in payload")
            head = self.model.module.bind({"params": self.model.params}).heads[self.head_name]
            dof_ids = tuple(int(x) for x in np.asarray(payload["dof_ids"]).reshape(-1))
            chunk_steps = tuple(float(x) for x in np.asarray(payload["chunk_steps"]).reshape(-1))
            kwargs["dof_ids"] = jnp.asarray(pad_dof_ids(dof_ids, head.max_dofs))[None]
            kwargs["chunk_steps"] = jnp.asarray(pad_chunk_steps(chunk_steps, head.max_horizon))[None]
        if "guide_input" in payload and payload["guide_input"] is not None:
            kwargs["guide_input"] = jnp.asarray(payload["guide_input"])

        self.rng, key = jax.random.split(self.rng)
        actions = self.model.sample_actions(
            observations=obs,
            tasks=self.task,
            unnormalization_statistics=None,
            head_name=self.head_name,
            rng=key,
            **kwargs,
        )
        actions = np.asarray(actions[0, -1])  # one batch, last window
        result = {"actions": actions}
        if self.is_xflow:
            padded_ids = np.asarray(kwargs["dof_ids"][0])
            result["dof_ids"] = padded_ids
        return result

    # ------------------------------------------------------------------
    # warmup
    # ------------------------------------------------------------------

    def warmup(self, n: int = 1):
        """Run ``n`` forward passes with dummy data to trigger JIT compilation."""
        self.task = self.model.example_batch["task"]
        batch = dict(self.model.example_batch)
        if self.is_xflow:
            head = self.model.module.bind({"params": self.model.params}).heads[self.head_name]
            batch["dof_ids"] = np.arange(head.max_dofs)
            batch["chunk_steps"] = np.arange(head.max_horizon, dtype=np.float32)
        for _ in range(n):
            self.step(batch)
