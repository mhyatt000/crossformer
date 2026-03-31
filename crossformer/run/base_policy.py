from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from webpolicy.base_policy import BasePolicy

from crossformer.model.components.heads.dof import pad_chunk_steps, pad_dof_ids
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.wrappers import _resize


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
