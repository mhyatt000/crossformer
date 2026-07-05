"""Fixed-batch action MSE eval callback for XFlow validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from einops import rearrange
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from rich import print
from rich.rule import Rule

from crossformer.embody import MASK_ID
from crossformer.run.train_step import lookup_guide
from crossformer.utils.callbacks.base import EvalContext, extract_bundled_actions
from crossformer.utils.callbacks.denorm import dof_name


@dataclass
class ValMSECallback:
    """Log fixed-batch action MSE for XFlow validation."""

    name: str = "val_mse"
    every: int = 0
    head_name: str = "action"
    sample_idx: int = 0
    print_sample: bool = True
    _eval_fns: dict[bool, Any] = field(default_factory=dict, init=False, repr=False)

    def __call__(self, ctx: EvalContext) -> dict[str, float]:
        batch = ctx.batch
        obs = batch["observation"]
        task = batch.get("task", {"pad_mask_dict": {}})
        actions, dof_ids, chunk_steps, view_ids, mask_act = extract_bundled_actions(batch, max_h=0)
        actions = jnp.asarray(actions)
        guide_input = lookup_guide(dict(batch), ctx.guide_keys) if ctx.use_guidance else None

        teacher, unguided, guided = self._eval_fn(ctx.model.module, ctx.use_guidance)(
            ctx.params,
            obs,
            task,
            actions,
            dof_ids,
            view_ids,
            chunk_steps,
            guide_input,
            ctx.rng,
        )

        out = {}
        samples = {}
        metrics, sample = self._collect(ctx, "teacher", teacher, actions, dof_ids, mask_act)
        out.update(metrics)
        samples["teacher"] = sample
        metrics, sample = self._collect(ctx, "unguided", unguided, actions, dof_ids, mask_act)
        out.update(metrics)
        samples["unguided"] = sample

        if ctx.use_guidance:
            metrics, sample = self._collect(ctx, "guided", guided, actions, dof_ids, mask_act)
            out.update(metrics)
            samples["guided"] = sample

        if self.print_sample:
            self._print_sample(ctx.step, samples)
        return out

    def _eval_fn(self, module: Any, use_guidance: bool) -> Any:
        fn = self._eval_fns.get(use_guidance)
        if fn is not None:
            return fn

        head_name = self.head_name

        @jax.jit
        def eval_fn(
            params: Any,
            obs: Any,
            task: Any,
            actions: Array,
            dof_ids: ArrayLike,
            view_ids: ArrayLike,
            chunk_steps: ArrayLike,
            guide_input: ArrayLike | None,
            rng: Any,
        ) -> Any:
            bound = module.bind({"params": params})
            transformer_outputs = bound.crossformer_transformer(
                obs,
                task,
                obs["timestep_pad_mask"],
                train=False,
            )
            teacher = bound.heads[head_name](
                transformer_outputs,
                time=jnp.ones((*actions.shape[:2], 1), dtype=jnp.float32),
                a_t=actions,
                dof_ids=dof_ids,
                chunk_steps=chunk_steps,
                view_ids=view_ids,
                train=False,
                guide_input=guide_input if use_guidance else None,
            )
            rng, key_no = jax.random.split(rng)
            unguided = bound.heads[head_name].predict_action(
                transformer_outputs,
                rng=key_no,
                dof_ids=dof_ids,
                chunk_steps=chunk_steps,
                view_ids=view_ids,
                train=False,
                guide_input=None,
            )
            guided = None
            if use_guidance:
                rng, key_yes = jax.random.split(rng)
                guided = bound.heads[head_name].predict_action(
                    transformer_outputs,
                    rng=key_yes,
                    dof_ids=dof_ids,
                    chunk_steps=chunk_steps,
                    view_ids=view_ids,
                    train=False,
                    guide_input=guide_input,
                )
            return teacher, unguided, guided

        self._eval_fns[use_guidance] = eval_fn
        return eval_fn

    def _collect(
        self,
        ctx: EvalContext,
        name: str,
        pred: ArrayLike,
        gt: ArrayLike,
        dof_ids: ArrayLike,
        mask_act: ArrayLike | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        ds_names = ctx.ds_names
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)
        gt = np.asarray(jax.device_get(gt), dtype=np.float32)
        dof_ids = np.asarray(jax.device_get(dof_ids))
        if mask_act is not None:
            mask_act = np.asarray(jax.device_get(mask_act), dtype=bool)
        if pred.ndim == 3:
            pred = rearrange(pred, "b w (h a) -> b w h a", h=gt.shape[2], a=gt.shape[3])
        pred = pred[:, 0, 0]
        gt = gt[:, 0, 0]

        pred_all = []
        gt_all = []
        pred_valid = []
        gt_valid = []
        per_dof: dict[str, list[float]] = {}

        for i, ds_name in enumerate(ds_names):
            pred_i = ctx.denorm.denormalize_slot(pred[i], dof_ids[i], ds_name)
            gt_i = ctx.denorm.denormalize_slot(gt[i], dof_ids[i], ds_name)
            # Per-DOF validity: prefer mask.act (respects per-joint vis gating);
            # fall back to just dropping pad slots.
            valid = mask_act[i] if mask_act is not None else np.asarray(dof_ids[i]) != MASK_ID

            pred_all.append(pred_i)
            gt_all.append(gt_i)
            pred_valid.append(pred_i[valid])
            gt_valid.append(gt_i[valid])

            for slot, dof_id in enumerate(np.asarray(dof_ids[i]).reshape(-1)):
                dof_id = int(dof_id)
                if dof_id == MASK_ID:
                    continue
                if mask_act is not None and not bool(mask_act[i, slot]):
                    continue
                err = float((pred_i[slot] - gt_i[slot]) ** 2)
                per_dof.setdefault(dof_name(dof_id), []).append(err)

        pred_all = np.stack(pred_all).reshape(-1)
        gt_all = np.stack(gt_all).reshape(-1)
        pred_valid = np.concatenate(pred_valid) if pred_valid else np.empty((0,), dtype=np.float32)
        gt_valid = np.concatenate(gt_valid) if gt_valid else np.empty((0,), dtype=np.float32)

        out = {
            f"{name}/all": self._mse(pred_all, gt_all),
            f"{name}/valid": self._mse(pred_valid, gt_valid),
            f"{name}/pred_min": float(pred_valid.min()),
            f"{name}/pred_max": float(pred_valid.max()),
            f"{name}/pred_mean": float(pred_valid.mean()),
            f"{name}/pred_std": float(pred_valid.std()),
            f"{name}/gt_min": float(gt_valid.min()),
            f"{name}/gt_max": float(gt_valid.max()),
            f"{name}/gt_mean": float(gt_valid.mean()),
            f"{name}/gt_std": float(gt_valid.std()),
        }
        for dname, errs in sorted(per_dof.items()):
            out[f"{name}/dof/{dname}"] = float(np.mean(np.asarray(errs, dtype=np.float32)))
        s = min(self.sample_idx, len(ds_names) - 1)
        pred_sample = pred_all.reshape(len(ds_names), -1)[s]
        gt_sample = gt_all.reshape(len(ds_names), -1)[s]
        sample = {
            "mse": self._mse(pred_sample, gt_sample),
            "dof_ids": np.asarray(dof_ids[s]).tolist(),
            "pred": np.asarray(pred_sample).round(3).tolist(),
            "gt": np.asarray(gt_sample).round(3).tolist(),
        }
        return out, sample

    def _print_sample(self, step: int, samples: Mapping[str, Mapping[str, Any]]) -> None:
        print(Rule(f"val mse step={step}"))
        for mode in ("teacher", "unguided", "guided"):
            if mode not in samples:
                continue
            sample = samples[mode]
            print(
                {
                    "mode": mode,
                    "mse": float(sample["mse"]),
                    "dof_ids": sample["dof_ids"],
                    "pred": sample["pred"],
                    "gt": sample["gt"],
                }
            )

    def _mse(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if pred.size == 0:
            return float("nan")
        return float(np.mean((pred - gt) ** 2))
