from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from crossformer.embody import DOF
from crossformer.run.train_step import lookup_guide
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer

JOINT_NAMES = tuple(f"j{i}" for i in range(7))
JOINT_IDS = tuple(DOF[name] for name in JOINT_NAMES)
JOINT_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(JOINT_IDS)}
RAST_NAMES = (*JOINT_NAMES, "gripper")
RAST_IDS = tuple(DOF[name] for name in RAST_NAMES)
RAST_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(RAST_IDS)}


def normalize_obs(obs, obs_keys):
    """Flatten selected lowdim inputs to (B, W, D)."""
    out = dict(obs)
    for key in obs_keys:
        x = out[key]
        if x.ndim == 2:
            out[key] = x[..., None]
        elif x.ndim > 3:
            out[key] = x.reshape(*x.shape[:2], -1)
    return out


def extract_bundled_actions(batch, max_h):
    """Extract bundled actions from grain embody pipeline."""
    del max_h
    actions = batch["act"]["base"]
    if actions.ndim == 3:
        actions = actions[:, None, :, :]
    bsz = actions.shape[0]
    horizon = actions.shape[2]
    dof_ids = batch["act"]["id"]
    chunk_steps = jnp.tile(jnp.arange(horizon, dtype=jnp.float32)[None], (bsz, 1))
    return actions, dof_ids, chunk_steps


def adapt_canonical_batch(act, flow, dof_id_to_idx):
    """Map bundled slot actions to a canonical DOF order."""
    base = np.asarray(act["base"], dtype=np.float32)
    dof_ids = np.asarray(act["id"])
    flow = np.asarray(flow, dtype=np.float32)
    if base.ndim == 3:
        base = base[:, None, :, :]
    if base.ndim != 4:
        raise ValueError(f"Expected act.base ndim 3 or 4, got {base.shape}")
    if flow.ndim != 5:
        raise ValueError(f"Expected flow ndim 5, got {flow.shape}")
    if dof_ids.ndim != 2:
        raise ValueError(f"Expected act.id ndim 2, got {dof_ids.shape}")
    if base.shape[0] != dof_ids.shape[0] or flow.shape[1] != dof_ids.shape[0]:
        raise ValueError(f"Batch mismatch: base={base.shape} flow={flow.shape} dof_ids={dof_ids.shape}")

    keep = []
    out_dim = len(dof_id_to_idx)
    base_joint = np.zeros((*base.shape[:-1], out_dim), dtype=np.float32)
    flow_joint = np.zeros((*flow.shape[:-1], out_dim), dtype=np.float32)
    for b, row in enumerate(dof_ids):
        has_target = False
        for src, dof_id in enumerate(row):
            dst = dof_id_to_idx.get(int(dof_id))
            if dst is None:
                continue
            has_target = True
            base_joint[b, ..., dst] = base[b, ..., src]
            flow_joint[:, b, ..., dst] = flow[:, b, ..., src]
        if has_target:
            keep.append(b)

    if not keep:
        return None, None
    keep = np.asarray(keep, dtype=np.int32)
    return {
        "act": {"base": base_joint[keep]},
        "predict": flow_joint[:, keep],
    }, keep


def adapt_viz_batch(act, flow):
    """Map bundled actions to canonical j0..j6 order for VizCallback."""
    return adapt_canonical_batch(act, flow, JOINT_ID_TO_IDX)


def adapt_rast_batch(act, flow):
    """Map bundled actions to canonical j0..j6+gripper order for RastCallback."""
    return adapt_canonical_batch(act, flow, RAST_ID_TO_IDX)


def denorm_canonical(arr: np.ndarray, denorm: ActionBatchDenormalizer, ds_name: str, dof_ids: np.ndarray) -> np.ndarray:
    """Denormalize canonical actions with explicit DOF ids."""
    arr = np.asarray(arr, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1])
    out = np.stack([denorm.denormalize_slot(row, dof_ids, ds_name) for row in flat], axis=0)
    return out.reshape(arr.shape)


@dataclass
class XFlowEvalCallbacks:
    """Callbacks and scheduling for eval."""

    hist_cb: Any
    chunk_cb: Any
    viz_cb: Any
    rast_cb: Any
    val_mse_cb: Any
    wandb_log: Callable[..., None]
    hist_every: int
    viz_every: int
    val_every: int
    eval_frames: int
    use_guidance: bool
    guide_keys: tuple[str, ...]


@dataclass
class XFlowEvalLoop:
    """Run eval callbacks from a separate loader."""

    loader: Iterable[Mapping[str, Any]]
    callbacks: XFlowEvalCallbacks
    obs_keys: tuple[str, ...]
    pred_rng: Any
    _it: Iterator[Mapping[str, Any]] | None = None

    def __call__(self, model, params, step: int, *, is_last: bool = False) -> None:
        need_hist = self.callbacks.hist_every > 0 and (step % self.callbacks.hist_every == 0 or is_last)
        need_viz = self.callbacks.viz_every > 0 and (step % self.callbacks.viz_every == 0 or is_last)
        need_val = self.callbacks.val_every > 0 and ((step + 1) % self.callbacks.val_every == 0)
        if not (need_hist or need_viz or need_val):
            return

        batch = self._next_batch()
        metrics = {}

        if need_hist or need_viz:
            rast_batches = [batch] if need_viz and self.callbacks.rast_cb is not None else None
            if rast_batches is not None:
                rast_batches = self._collect_batches(batch, self.callbacks.eval_frames)
            metrics.update(
                self._predict_metrics(
                    model,
                    params,
                    batch,
                    need_hist=need_hist,
                    need_viz=need_viz,
                    rast_batches=rast_batches,
                )
            )

        val_metrics = self.callbacks.val_mse_cb.every(
            model,
            params,
            batch,
            step,
            self.callbacks.val_every,
            self.pred_rng,
            self.callbacks.use_guidance,
        )
        if val_metrics is not None:
            metrics.update(val_metrics)

        if metrics:
            self.callbacks.wandb_log(metrics, step=step)

    def _next_batch(self) -> Mapping[str, Any]:
        if self._it is None:
            self._it = iter(self.loader)
        try:
            batch = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            batch = next(self._it)
        batch = dict(batch)
        batch["observation"] = normalize_obs(batch["observation"], self.obs_keys)
        return batch

    def _collect_batches(self, first_batch, frames: int) -> list[Mapping[str, Any]]:
        out = [first_batch]
        seen = int(np.asarray(first_batch["act"]["id"]).shape[0])
        while seen < frames:
            batch = self._next_batch()
            out.append(batch)
            seen += int(np.asarray(batch["act"]["id"]).shape[0])
        return out

    def _predict_metrics(self, model, params, batch, *, need_hist: bool, need_viz: bool, rast_batches=None) -> dict[str, Any]:
        obs = batch["observation"]
        task = batch.get("task", {"pad_mask_dict": {}})
        _, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h=0)
        guide_input = lookup_guide(batch, self.callbacks.guide_keys) if self.callbacks.use_guidance else None

        bound = model.module.bind({"params": params})
        transformer_outputs = bound.crossformer_transformer(
            obs,
            task,
            obs["timestep_pad_mask"],
            train=False,
        )
        pred = bound.heads["xflow"].predict_action(
            transformer_outputs,
            rng=self.pred_rng,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            train=False,
            guide_input=guide_input,
        )
        pred_np = jax.device_get(pred)
        out = {}

        if need_hist:
            pred_flat = pred_np.reshape(pred_np.shape[0], pred_np.shape[1], -1)
            out.update(self.callbacks.hist_cb(batch, {"predict": pred_flat}))
            chunk_imgs = self.callbacks.chunk_cb(batch, {"predict": pred_flat})
            for k, v in chunk_imgs.items():
                out[f"action_chunks/{k}"] = v

        if not need_viz:
            return out

        pred_flow = bound.heads["xflow"].predict_action(
            transformer_outputs,
            rng=self.pred_rng,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            train=False,
            guide_input=guide_input,
            accumulate=True,
        )
        pred_flow_np = jax.device_get(pred_flow)
        viz_batch, _ = adapt_viz_batch(batch["act"], pred_flow_np)
        if viz_batch is not None:
            frames = self.callbacks.viz_cb(viz_batch)
            out["flow_pca/video"] = wandb.Video(
                np.moveaxis(frames, -1, 1),
                fps=self.callbacks.viz_cb.fps,
            )

        if self.callbacks.rast_cb is None:
            return out

        cam_videos = self._rast_videos(model, params, rast_batches or [batch])
        out.update(cam_videos)
        return out

    def _rast_videos(self, model, params, batches) -> dict[str, Any]:
        per_cam: list[list[np.ndarray]] | None = None
        frames_left = self.callbacks.eval_frames
        for batch in batches:
            if frames_left <= 0:
                break
            pred_flow_np = self._predict_flow(model, params, batch)
            rast_batch, rast_keep = adapt_rast_batch(batch["act"], pred_flow_np)
            if rast_batch is None:
                continue

            ds_names = self.callbacks.chunk_cb.denorm.decode_dataset_names(
                jax.device_get(batch["info"]["dataset_name"]),
            )
            kept = np.asarray(rast_keep, dtype=np.int32)
            n_take = min(frames_left, len(kept))
            for local_idx in range(n_take):
                ds_name = ds_names[int(kept[local_idx])]
                chunk = rast_batch["predict"][-1, local_idx]
                if chunk.ndim == 3:
                    chunk = chunk[0]
                chunk = denorm_canonical(chunk, self.callbacks.chunk_cb.denorm, ds_name, np.asarray(RAST_IDS))
                traj_frames = self.callbacks.rast_cb.render_trajectory(chunk)
                if per_cam is None:
                    per_cam = [[] for _ in range(len(traj_frames))]
                for ci, frame in enumerate(traj_frames):
                    per_cam[ci].append(frame)
                frames_left -= 1
                if frames_left <= 0:
                    break

        if per_cam is None:
            return {}
        out = {}
        fps = getattr(self.callbacks.viz_cb, "fps", 10)
        for ci, frames in enumerate(per_cam):
            video = np.stack(frames, axis=0)
            out[f"rast/cam_{ci}"] = wandb.Video(np.moveaxis(video, -1, 1), fps=fps)
        return out

    def _predict_flow(self, model, params, batch) -> np.ndarray:
        obs = batch["observation"]
        task = batch.get("task", {"pad_mask_dict": {}})
        _, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h=0)
        guide_input = lookup_guide(batch, self.callbacks.guide_keys) if self.callbacks.use_guidance else None
        bound = model.module.bind({"params": params})
        transformer_outputs = bound.crossformer_transformer(
            obs,
            task,
            obs["timestep_pad_mask"],
            train=False,
        )
        pred_flow = bound.heads["xflow"].predict_action(
            transformer_outputs,
            rng=self.pred_rng,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            train=False,
            guide_input=guide_input,
            accumulate=True,
        )
        return jax.device_get(pred_flow)
