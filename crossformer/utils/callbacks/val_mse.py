from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from einops import rearrange
import jax
import jax.numpy as jnp
import numpy as np
from rich import print
from rich.rule import Rule

from crossformer.embody import DOF, MASK_ID
from crossformer.run.train_step import lookup_guide
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer


@dataclass
class ValMSECallback:
    """Log fixed-batch action MSE for XFlow validation."""

    stats: Mapping[str, Any]
    head_name: str = "xflow"
    ds_key: tuple[str, ...] = ("info", "dataset_name")
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")
    sample_idx: int = 0
    print_sample: bool = True
    _eval_fns: dict[bool, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.denorm = ActionBatchDenormalizer(self.stats)

    def every(
        self,
        model,
        params,
        batch: Mapping[str, Any],
        step: int,
        log_every: int,
        rng,
        use_guidance: bool,
    ) -> dict[str, float] | None:
        if log_every <= 0 or (step + 1) % log_every != 0:
            return None
        return self(model, params, batch, step, rng, use_guidance)

    def __call__(
        self,
        model,
        params,
        batch: Mapping[str, Any],
        step: int,
        rng,
        use_guidance: bool,
    ) -> dict[str, float]:
        obs = batch["observation"]
        task = batch.get("task", {"pad_mask_dict": {}})
        actions = batch["act"]["base"]
        if actions.ndim == 3:
            actions = actions[:, None, :, :]
        dof_ids = batch["act"]["id"]
        chunk_steps = jnp.tile(jnp.arange(actions.shape[2], dtype=jnp.float32)[None], (actions.shape[0], 1))
        ds_names = self.denorm.decode_dataset_names(jax.device_get(self._get(batch, self.ds_key)))
        guide_input = lookup_guide(batch, self.guide_keys) if use_guidance else None

        teacher, unguided, guided = self._eval_fn(model.module, use_guidance)(
            params,
            obs,
            task,
            actions,
            dof_ids,
            chunk_steps,
            guide_input,
            rng,
        )

        out = {}
        samples = {}
        metrics, sample = self._collect("teacher", teacher, actions, dof_ids, ds_names)
        out.update(metrics)
        samples["teacher"] = sample
        metrics, sample = self._collect("unguided", unguided, actions, dof_ids, ds_names)
        out.update(metrics)
        samples["unguided"] = sample

        if use_guidance:
            metrics, sample = self._collect("guided", guided, actions, dof_ids, ds_names)
            out.update(metrics)
            samples["guided"] = sample

        if self.print_sample:
            self._print_sample(step, samples)
        return out

    def _eval_fn(self, module, use_guidance: bool):
        fn = self._eval_fns.get(use_guidance)
        if fn is not None:
            return fn

        head_name = self.head_name

        @jax.jit
        def eval_fn(params, obs, task, actions, dof_ids, chunk_steps, guide_input, rng):
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
                train=False,
                guide_input=guide_input if use_guidance else None,
            )
            rng, key_no = jax.random.split(rng)
            unguided = bound.heads[head_name].predict_action(
                transformer_outputs,
                rng=key_no,
                dof_ids=dof_ids,
                chunk_steps=chunk_steps,
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
                    train=False,
                    guide_input=guide_input,
                )
            return teacher, unguided, guided

        self._eval_fns[use_guidance] = eval_fn
        return eval_fn

    def _collect(
        self,
        name: str,
        pred,
        gt,
        dof_ids,
        ds_names: list[str],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        pred = np.asarray(jax.device_get(pred), dtype=np.float32)
        gt = np.asarray(jax.device_get(gt), dtype=np.float32)
        dof_ids = np.asarray(jax.device_get(dof_ids))
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
            pred_i = self.denorm.denormalize_slot(pred[i], dof_ids[i], ds_name)
            gt_i = self.denorm.denormalize_slot(gt[i], dof_ids[i], ds_name)
            valid = np.asarray(dof_ids[i]) != MASK_ID

            pred_all.append(pred_i)
            gt_all.append(gt_i)
            pred_valid.append(pred_i[valid])
            gt_valid.append(gt_i[valid])

            for slot, dof_id in enumerate(np.asarray(dof_ids[i]).reshape(-1)):
                dof_id = int(dof_id)
                if dof_id == MASK_ID:
                    continue
                dof_name = self._dof_name(dof_id)
                err = float((pred_i[slot] - gt_i[slot]) ** 2)
                per_dof.setdefault(dof_name, []).append(err)

        pred_all = np.stack(pred_all).reshape(-1)
        gt_all = np.stack(gt_all).reshape(-1)
        pred_valid = np.concatenate(pred_valid) if pred_valid else np.empty((0,), dtype=np.float32)
        gt_valid = np.concatenate(gt_valid) if gt_valid else np.empty((0,), dtype=np.float32)

        out = {
            f"val_mse/{name}/all": self._mse(pred_all, gt_all),
            f"val_mse/{name}/valid": self._mse(pred_valid, gt_valid),
            f"val_mse/{name}/pred_min": float(pred_valid.min()),
            f"val_mse/{name}/pred_max": float(pred_valid.max()),
            f"val_mse/{name}/pred_mean": float(pred_valid.mean()),
            f"val_mse/{name}/pred_std": float(pred_valid.std()),
            f"val_mse/{name}/gt_min": float(gt_valid.min()),
            f"val_mse/{name}/gt_max": float(gt_valid.max()),
            f"val_mse/{name}/gt_mean": float(gt_valid.mean()),
            f"val_mse/{name}/gt_std": float(gt_valid.std()),
        }
        for dof_name, errs in sorted(per_dof.items()):
            out[f"val_mse/{name}/dof/{dof_name}"] = float(np.mean(np.asarray(errs, dtype=np.float32)))
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

    def _get(self, batch: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        cur = batch
        for key in path:
            cur = cur[key]
        return cur

    def _mse(self, pred: np.ndarray, gt: np.ndarray) -> float:
        if pred.size == 0:
            return float("nan")
        return float(np.mean((pred - gt) ** 2))

    def _dof_name(self, dof_id: int) -> str:
        for name, idx in DOF.items():
            if idx == dof_id:
                return name
        return f"dof_{dof_id}"
