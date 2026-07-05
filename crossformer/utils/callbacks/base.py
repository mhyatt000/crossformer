"""Eval callback contract: ``EvalCallback`` protocol + shared ``EvalContext``.

Every eval-time callback implements the same interface: a wandb ``name``
namespace, an ``every`` schedule (0 = disabled), and ``__call__(ctx)``
returning a flat dict of wandb-loggables. The loop (``EvalLoop``) prefixes
each key with ``{name}/`` and logs once per tick.

``EvalContext`` owns the model forward passes: ``pred`` and ``pred_flow``
are cached properties, so however many callbacks are due on a tick, each
prediction kind is computed at most once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from crossformer.model.components.heads.dof import CHUNK_PAD
from crossformer.run.train_step import lookup_guide
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer


def getpath(tree: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Fetch a nested value by key path; raises KeyError on a missing key."""
    cur: Any = tree
    for key in path:
        cur = cur[key]
    return cur


def maybe_getpath(tree: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
    """Fetch a nested value by key path; None when any key is missing."""
    cur: Any = tree
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def flatten_obs(
    obs: Mapping[str, Any],
    obs_keys: tuple[str, ...],
    *,
    view_mask: ArrayLike | None = None,
) -> dict[str, Any]:
    """Flatten selected lowdim inputs to (B, W, D).

    When ``view_mask`` (data-side ``mask.view``, which stacked slots hold a real
    camera vs zero padding) is given and not already present, inject it as the
    ``view_mask`` observation the stacked tokenizer reads. A bare ``(B, V)`` mask
    is broadcast over the window so every observation leaf shares the model's
    horizon (the transformer asserts this per leaf).
    """
    out = dict(obs)
    for key in obs_keys:
        x = out[key]
        if x.ndim == 2:
            out[key] = x[..., None]
        elif x.ndim > 3:
            out[key] = x.reshape(*x.shape[:2], -1)
    if view_mask is not None and "view_mask" not in out:
        vm = jnp.asarray(view_mask)
        if vm.ndim == 2:
            horizon = int(out["timestep_pad_mask"].shape[1])
            vm = jnp.broadcast_to(vm[:, None, :], (vm.shape[0], horizon, vm.shape[-1]))
        out["view_mask"] = vm
    return out


def extract_bundled_actions(batch: Mapping[str, Any], max_h: int) -> tuple[Array, Array, Array, Array, Array | None]:
    """Extract bundled actions from grain embody pipeline.

    Returns ``(actions, dof_ids, chunk_steps, view_ids, mask_act)`` where
    ``view_ids`` is the per-slot camera id (``act.view``; zeros when absent) and
    ``mask_act`` is the per-slot supervision mask (``mask.act``; None when absent).
    ``mask.horizon`` is folded into ``chunk_steps`` using CHUNK_PAD so invalid
    future action steps are masked by the head's existing query mask.
    """
    del max_h
    actions = batch["act"]["base"]
    if actions.ndim == 3:
        actions = actions[:, None, :, :]
    bsz = actions.shape[0]
    horizon = actions.shape[2]
    dof_ids = batch["act"]["id"]
    view_ids = batch["act"].get("view")
    if view_ids is None:
        view_ids = jnp.zeros_like(dof_ids)
    mask_act = batch.get("mask", {}).get("act")
    chunk_steps = jnp.tile(jnp.arange(horizon, dtype=jnp.float32)[None], (bsz, 1))
    horizon_mask = batch.get("mask", {}).get("horizon")
    if horizon_mask is not None:
        horizon_mask = jnp.asarray(horizon_mask, dtype=bool)
        if horizon_mask.ndim == 1:
            horizon_mask = horizon_mask[None]
        horizon_mask = horizon_mask[:, :horizon]
        chunk_steps = jnp.where(horizon_mask, chunk_steps, CHUNK_PAD)
    return actions, dof_ids, chunk_steps, view_ids, mask_act


@runtime_checkable
class EvalCallback(Protocol):
    """Contract every eval callback implements."""

    name: str  # wandb namespace, e.g. "rast", "flow_pca"
    every: int  # 0 = disabled; scheduling lives HERE, once

    def __call__(self, ctx: EvalContext) -> dict[str, Any]: ...  # wandb-loggables


@dataclass
class EvalContext:
    """Shared, lazily-computed prediction state for one eval tick."""

    model: Any
    params: Any
    rng: Any
    step: int
    batch: Mapping[str, Any]  # already flattened
    denorm: ActionBatchDenormalizer
    stats: Any = None  # raw DatasetStatistics (has .unnormalize)
    head_name: str = "action"
    use_guidance: bool = False
    guide_keys: tuple[str, ...] = ()
    # Factory for a context over a fresh eval batch; wired by the loop so
    # ``stream`` can pull more data (e.g. rast rendering many samples).
    next_ctx: Callable[[], EvalContext] | None = field(default=None, repr=False)

    @property
    def batch_size(self) -> int:
        return int(np.asarray(self.batch["act"]["id"]).shape[0])

    @cached_property
    def ds_names(self) -> list[str]:
        return self.denorm.decode_dataset_names(jax.device_get(self.batch["info"]["dataset_name"]))

    @cached_property
    def pred(self) -> np.ndarray:
        """predict_action, computed once."""
        return self._predict(accumulate=False)

    @cached_property
    def pred_flow(self) -> np.ndarray:
        """predict_action(accumulate=True), computed once."""
        return self._predict(accumulate=True)

    def stream(self, n_frames: int) -> Iterator[EvalContext]:
        """Yield self, then fresh-batch contexts until ~n_frames samples seen."""
        yield self
        seen = self.batch_size
        while seen < n_frames and self.next_ctx is not None:
            ctx = self.next_ctx()
            yield ctx
            seen += ctx.batch_size

    @cached_property
    def _bound(self) -> Any:
        return self.model.module.bind({"params": self.params})

    @cached_property
    def _transformer_outputs(self) -> Any:
        obs = self.batch["observation"]
        task = self.batch.get("task", {"pad_mask_dict": {}})
        return self._bound.crossformer_transformer(obs, task, obs["timestep_pad_mask"], train=False)

    def _predict(self, *, accumulate: bool) -> np.ndarray:
        _, dof_ids, chunk_steps, view_ids, _ = extract_bundled_actions(self.batch, max_h=0)
        guide_input = lookup_guide(dict(self.batch), self.guide_keys) if self.use_guidance else None
        pred = self._bound.heads[self.head_name].predict_action(
            self._transformer_outputs,
            rng=self.rng,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            view_ids=view_ids,
            train=False,
            guide_input=guide_input,
            accumulate=accumulate,
        )
        return jax.device_get(pred)
