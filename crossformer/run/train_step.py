from __future__ import annotations

from functools import partial
from typing import Any, Callable

from einops import rearrange
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import optax

from crossformer.utils.mytyping import Params
from crossformer.utils.tree.core import flat


def lookup_guide(batch: dict, keys: tuple[str, ...]) -> jnp.ndarray:
    """Look up guide keys from batch and concat along last dim."""
    f = flat(batch)
    parts = []
    for k in keys:
        if k not in f:
            raise KeyError(f"guide key {k!r} not found in batch. available: {sorted(f)}")
        v = f[k]
        if v.ndim > 3:
            v = rearrange(v, "b w ... -> b w (...)")
        parts.append(v)
    return jnp.concatenate(parts, axis=-1)


def make_train_step(
    module: Any,
    lr_callable: float | Callable[[int], float] = 1e-3,
    param_norm_callable: Callable[[Params], float] = optax.global_norm,
    dof_weights: ArrayLike | None = None,
) -> Callable:
    """Build a compiled train step using bundled action format.

    Expects actions as (B, W, H, max_a) with dof_ids from act.id.
    No per-head loop — single unified action space.

    Args:
        module: CrossFormerModel module.
        lr_callable: learning rate schedule fn(step) -> lr, or constant float.
        param_norm_callable: fn(params) -> scalar norm (respects frozen keys).
    Returns:
        Compiled train_step(state, obs, task, pad_mask, actions, dof_ids, chunk_steps, guide_input).
    """
    if not callable(lr_callable):
        _lr = lr_callable
        lr_callable = lambda _: _lr

    @partial(jax.jit, static_argnames=("train",))
    def train_step(
        state: Any,
        obs: Any,
        task: Any,
        pad_mask: ArrayLike,
        actions: ArrayLike,
        dof_ids: ArrayLike,
        chunk_steps: ArrayLike,
        guide_input: ArrayLike | None = None,
        view_ids: ArrayLike | None = None,
        mask_act: ArrayLike | None = None,
        train: bool = True,
    ) -> tuple[Any, dict[str, Any]]:
        """Bundled train step: fwd transformer, compute loss on unified action block.

        Args:
            state: TrainState.
            obs: observation dict.
            task: task dict.
            pad_mask: (B, W) timestep pad mask.
            actions: (B, W, H, max_a) padded actions from act.base.
            dof_ids: (B, max_a) DOF vocab IDs from act.id (MASK_ID=0 for padding).
            chunk_steps: (B, H) temporal positions — just arange(H) for now.
            guide_input: optional (B, S, D) guidance signal.
            view_ids: optional (B, max_a) per-slot camera id from act.view (zeros when None).
            mask_act: optional (B, max_a) per-slot supervision mask from mask.act.
            train: bool.

        Returns:
            (state, update_info).
        """
        rng = jax.random.fold_in(state.rng, state.step)
        params = state.model.params

        def _total_loss(params: Any) -> tuple[Array, dict[str, Array]]:
            bound = module.bind({"params": params}, rngs={"dropout": rng})
            transformer_outputs = bound.crossformer_transformer(obs, task, pad_mask, train=train)

            loss, metrics = bound.heads["action"].loss(
                transformer_outputs,
                actions,
                dof_ids,
                chunk_steps,
                view_ids=view_ids,
                train=train,
                guide_input=guide_input,
                mask_act=mask_act,
                dof_weights=dof_weights,
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(_total_loss, has_aux=True)(params)
        updates, _ = state.tx.update(grads, state.opt_state, params)
        update_info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": param_norm_callable(params),
            "learning_rate": lr_callable(state.step),
            **metrics,
        }
        _, new_rng = jax.random.split(state.rng)
        state = state.apply_gradients(grads=grads, rng=new_rng)
        return state, update_info

    return train_step
