from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax


def make_train_step(module, lr, guide_module=None):
    """Build a compiled train step using bundled action format.

    Expects actions as (B, W, H, max_a) with dof_ids from act.id.
    No per-head loop — single unified action space.

    Args:
        module: CrossFormerModel module.
        lr: optimizer learning rate.
        guide_module: optional TokenGuidance module.

    Returns:
        Compiled train_step(state, obs, task, pad_mask, actions, dof_ids, chunk_steps, guide_input).
    """

    @partial(jax.jit, static_argnames=("train",))
    def train_step(state, obs, task, pad_mask, actions, dof_ids, chunk_steps, guide_input=None, train=True):
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
            train: bool.

        Returns:
            (state, update_info).
        """
        rng = jax.random.fold_in(state.rng, state.step)
        params = state.model.params

        def _total_loss(params):
            model_params = params["model"] if guide_module is not None else params

            bound = module.bind({"params": model_params}, rngs={"dropout": rng})
            transformer_outputs = bound.crossformer_transformer(obs, task, pad_mask, train=train)

            guidance_tokens = None
            if guide_module is not None and guide_input is not None:
                guidance_tokens = guide_module.apply(
                    {"params": params["guide"]},
                    guide_input,
                    deterministic=not train,
                )

            loss, metrics = bound.heads["xflow"].loss(
                transformer_outputs,
                actions,
                dof_ids,
                chunk_steps,
                train=train,
                guidance_tokens=guidance_tokens,
            )
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(_total_loss, has_aux=True)(params)
        updates, _ = state.tx.update(grads, state.opt_state, params)
        update_info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": optax.global_norm(params),
            "learning_rate": jnp.asarray(lr),
            **metrics,
        }
        _, new_rng = jax.random.split(state.rng)
        state = state.apply_gradients(grads=grads, rng=new_rng)
        return state, update_info

    return train_step
