from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from webpolicy.base_policy import BasePolicy

from crossformer.data.grain import metadata
from crossformer.data.grain.embody import build_action_norm_mask
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
        head_name: str = "action",
        guide_keys: tuple[str, ...] = ("action.position", "action.orientation"),
        use_guidance: bool = True,
        flow_steps: int | None = None,
    ):
        self.model: CrossFormerModel = CrossFormerModel.load_pretrained(path, step=step)
        self.params = self.model.params
        self.head_name = head_name
        self.model.module.heads[head_name].flow_steps = (
            flow_steps if flow_steps is not None else self.model.heads[head_name].flow_steps
        )
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
        B = jnp.asarray(obs["timestep_pad_mask"]).shape[0]
        task = payload.get("task", jax.tree.map(lambda x: x[:B], self.model.example_batch["task"]))

        # print(obs['timestep_pad_mask'])
        # print({k:v for k,v in obs.items() if 'proprio' in k})

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
        out = {"actions": jax.device_get(pred), "dof_ids": jax.device_get(dof_ids)}
        # print(out['actions'][...,:8])
        # print(out['dof_ids'][:,:8])
        return out


from crossformer.embody import DOF, MASK_ID


def slots_to_action_dict(actions: np.ndarray, dof_ids: np.ndarray) -> dict[str, np.ndarray]:
    xs = np.asarray(actions, dtype=np.float32)
    ids = np.asarray(dof_ids)
    if ids.ndim == 2:
        if xs.shape[0] != ids.shape[0]:
            raise ValueError(f"Batch mismatch: actions {xs.shape}, dof_ids {ids.shape}")
        rows = [slots_to_action_dict(xs[b], ids[b]) for b in range(xs.shape[0])]
        keys = sorted({k for row in rows for k in row})
        return {k: np.stack([row[k] for row in rows], axis=0) for k in keys}
    ids = ids.reshape(-1)
    n = min(xs.shape[-1], ids.shape[0])
    xs = xs[..., :n]
    ids = ids[:n]

    out: dict[str, list[tuple[int, np.ndarray]]] = {
        "joints": [],
        "gripper": [],
        "position": [],
        "orientation": [],
    }

    for slot, dof_id in enumerate(ids):
        dof_id = int(dof_id)
        if dof_id == MASK_ID:
            continue

        if DOF["j0"] <= dof_id <= DOF["j6"]:
            out["joints"].append((dof_id - DOF["j0"], xs[..., slot]))
        elif dof_id == DOF["gripper"]:
            out["gripper"].append((0, xs[..., slot]))
        elif DOF["ee_x"] <= dof_id <= DOF["ee_z"]:
            out["position"].append((dof_id - DOF["ee_x"], xs[..., slot]))
        elif DOF["ee_rx"] <= dof_id <= DOF["ee_rz"]:
            out["orientation"].append((dof_id - DOF["ee_rx"], xs[..., slot]))

    return {k: np.stack([v for _, v in sorted(vals)], axis=-1) for k, vals in out.items() if vals}


def action_dict_to_slots(action: dict[str, np.ndarray], dof_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}
    ids = np.asarray(dof_ids)
    if ids.ndim == 2:
        if not xs:
            return np.zeros(ids.shape, dtype=np.float32), ids
        if any(v.shape[0] != ids.shape[0] for v in xs.values()):
            shapes = {k: v.shape for k, v in xs.items()}
            raise ValueError(f"Batch mismatch: action {shapes}, dof_ids {ids.shape}")
        rows = [action_dict_to_slots({k: v[b] for k, v in xs.items()}, ids[b])[0] for b in range(ids.shape[0])]
        return np.stack(rows, axis=0), ids
    ids = ids.reshape(-1)
    out_shape = (*next(iter(xs.values())).shape[:-1], ids.shape[0]) if xs else (ids.shape[0],)
    out = np.zeros(out_shape, dtype=np.float32)

    for slot, dof_id in enumerate(ids):
        dof_id = int(dof_id)
        if dof_id == MASK_ID:
            continue

        if DOF["j0"] <= dof_id <= DOF["j6"] and "joints" in xs:
            out[..., slot] = xs["joints"][..., dof_id - DOF["j0"]]
        elif dof_id == DOF["gripper"] and "gripper" in xs:
            out[..., slot] = xs["gripper"][..., 0]
        elif DOF["ee_x"] <= dof_id <= DOF["ee_z"] and "position" in xs:
            out[..., slot] = xs["position"][..., dof_id - DOF["ee_x"]]
        elif DOF["ee_rx"] <= dof_id <= DOF["ee_rz"] and "orientation" in xs:
            out[..., slot] = xs["orientation"][..., dof_id - DOF["ee_rx"]]

    return out, ids


class ActionDenormWrapper(PolicyWrapper):
    """Wraps a policy and denormalizes the 'actions' field in the result.

    Handles step shapes (B, W, H, A) and flow shapes (T, B, W, H, A) — B is
    at axis 0 for step and axis 1 for flow (ndim==5).  Denorm is applied
    per-sample using dof_ids and dataset_name (fixed at construction time).
    """

    def __init__(self, inner, stats, embodiment):  # , dataset_name: str):
        self.inner = inner
        self._denorm = ActionBatchDenormalizer(stats)
        # self.dataset_name = dataset_name
        self.embodiment = embodiment
        self.stats = stats

        eb = self.unwrapped().model.example_batch
        fake_action = prop = {k.replace("proprio_", ""): v for k, v in eb["observation"].items() if "proprio" in k}
        self.norm_mask: dict = build_action_norm_mask(fake_action, self.embodiment)

    def step(self, payload: dict, **kwargs) -> dict:
        result = dict(self.inner.step(payload, **kwargs))
        result["actions"] = slots_to_action_dict(result["actions"], result["dof_ids"])
        result = self.denorm_new(result)

        # B = np.asarray(result["dof_ids"]).shape[0]
        # ds_names = [self.dataset_name] * B
        # result["actions"] = self._denorm_actions(result["actions"], result["dof_ids"], ds_names)
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

    def denorm_new(self, x: dict):
        # 8. normalize action when present; debug server payloads may omit GT actions.
        # eb = self.unwrapped().model.example_batch
        # print(spec(eb))
        # self.norm_mask: dict = build_action_norm_mask(eb["action"], embodiment)

        actk = "actions"  # 'action'

        def denorm_all(x: dict, stats: metadata.DatasetStatistics):
            # we do this to fix jax.tree.map keyerror of normalize_tree
            norm_mask = {k: self.norm_mask[k] for k in x[actk]}
            stats_action = {k: stats.action[k] for k in x[actk]}
            denorm = partial(metadata.normalize_tree, mask=norm_mask, inv=True)
            # if has_action and self.norm_action:
            x[actk] = denorm(x[actk], stats_action)
            # x["observation"]["proprio"] = denorm(x["observation"]["proprio"], stats.proprio)
            return x

        return denorm_all(x, self.stats)


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
