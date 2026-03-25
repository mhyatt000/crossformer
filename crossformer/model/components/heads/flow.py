"""Flow matching action heads."""

from __future__ import annotations

from functools import partial
from typing import Callable

from einops import rearrange
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.diffusion import create_diffusion_model
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.mytyping import PRNGKey
from crossformer.utils.type_checking import Action, Batched, BWC, One, Scalar, Windowed

from .base import ContinuousActionHead
from .losses import continuous_loss, sample_tau


class FlowMatchingActionHead(ContinuousActionHead):
    """Flow-matching head that predicts conditional action velocities."""

    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    use_layer_norm: bool = True
    flow_steps: int = 10
    base_std: float = 1.0

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        self.flow_model = create_diffusion_model(
            self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

    def _embed(self, transformer_outputs: dict[str, TokenGroup], train: bool) -> jax.Array:
        """Extract embeddings from transformer outputs."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            "Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )
        print(transformer_outputs.keys())
        print("token_group.tokens", token_group.tokens.shape)

        if self.pool_strategy == "use_map":
            return self.map_head(token_group, train=train)[:, :, 0]
        if self.pool_strategy == "mean":
            return token_group.tokens.mean(axis=-2)
        if self.pool_strategy == "pass":
            return token_group.tokens
        raise ValueError(f"{self.pool_strategy} not implemented!")

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        a_t: ArrayLike | None = None,
        train: bool = True,
    ) -> jax.Array:
        """Predict action velocities."""
        embeddings = self._embed(transformer_outputs, train=train)
        print("embeddings", embeddings.shape)

        if (time is None or a_t is None) and not self.is_initializing():
            raise ValueError("Must provide time and current action when calling flow head")
        if self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            a_t = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                dtype=jnp.float32,
            )

        if a_t.ndim == 4:
            a_t = rearrange(a_t, "b w h a -> b w (h a)")

        return self.flow_model(embeddings, a_t, time, train=train)

    def flow_loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: Batched[Windowed[Scalar]],
        action_pad_mask: ArrayLike,
        action_head_mask: Batched[Scalar] | None = None,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute flow matching loss."""
        if self.constrain_loss_dims:
            actions: BWC[Action] = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        rng = self.make_rng("dropout")
        base_key, time_key = jax.random.split(rng)
        base = self.base_std * jax.random.normal(base_key, actions_flat.shape)

        # Sample time from beta distribution
        time: Batched[Windowed[One]] = sample_tau(time_key, shape=(*actions_flat.shape[:2], 1), s=0.99)

        blended = time * actions_flat + (1.0 - time) * base
        target = actions_flat - base

        pred = self(
            transformer_outputs,
            time=time,
            a_t=blended,
            train=train,
        )

        if action_head_mask is None:
            action_head_mask: Batched[Scalar] = jnp.ones(pred.shape[0], dtype=bool)

        mask = rearrange(
            (timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]),
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
        return loss, metrics

    def loss(self, *args, **kwargs) -> tuple[Array, dict[str, Array]]:
        """Compute loss (delegates to flow_loss)."""
        return self.flow_loss(*args, **kwargs)

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        *args,
        sample_shape: tuple[int, ...] = (1,),
        accumulate: bool = False,
        **kwargs,
    ) -> jax.Array:
        """Predict actions by solving ODE through flow.

        Args:
            accumulate: If True, return full trajectory [F+1, B, W, H, A].
                If False (default), return only final prediction [B, W, H, A].
        """
        module, variables = self.unbind()

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            tokens = transformer_outputs[self.readout_key].tokens
            batch_size, window_size = tokens.shape[:2]
            a_t = self.base_std * jax.random.normal(
                key,
                (
                    batch_size,
                    window_size,
                    self.action_horizon * self.action_dim,
                ),
            )

            dt = 1.0 / max(self.flow_steps, 1)

            def scan_fn(a_t, step):
                t = (step + 0.5) * dt
                time = jnp.full((*a_t.shape[:2], 1), t, dtype=a_t.dtype)
                velocity = module.apply(
                    variables,
                    transformer_outputs,
                    time,
                    a_t,
                    train=train,
                )
                updated = a_t + dt * velocity
                if self.clip_pred:
                    updated = jnp.clip(updated, -self.max_action, self.max_action)
                return updated, updated if accumulate else ()

            steps = jnp.arange(self.flow_steps)
            a_t, history = jax.lax.scan(scan_fn, a_t, steps)

            if accumulate:
                source = jnp.concatenate([a_t[None, ...], history], axis=0)
                fmt = "f b w (h a) -> f b w h a"
            else:
                source = a_t
                fmt = "b w (h a) -> b w h a"

            actions = rearrange(
                source,
                fmt,
                h=self.action_horizon,
                a=self.action_dim,
            )
            if self.clip_pred:
                actions = jnp.clip(actions, -self.max_action, self.max_action)
            return actions

        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        samples = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        samples = samples.reshape(sample_shape + samples.shape[1:])
        return samples


def cart_flow_loss(qt: jax.Array, q1: jax.Array, t: jax.Array, qvh: jax.Array, fk: Callable, J_fn: Callable):
    """Compute cartesian space flow loss.

    Args:
        q: denotes joint space
        x: denotes Cartesian/task space (e.g., 3 or 6 DOF)
        D: DOF dimension
        M: Cartesian/task dimension
    """
    J = J_fn(qt)
    xvh = jnp.einsum("...md,...d->...m", J, qvh)
    xt = fk(qt)
    x1 = fk(q1)
    dx = x1 - xt
    xv = dx / (dt := (1.0 - t))

    return xvh, xv


def _cart_flow_adjustor(
    batch,
    flows,
    timestep_pad_mask,
    action_pad_mask,
    action_head_mask,
    fk: Callable,
    J_fn: Callable,
):
    """Apply cartesian space adjustments to flow predictions."""
    q1 = batch["action"]
    qt = rearrange(flows["blended"], "b w (h a) -> b w h a", h=q1.shape[-2], a=q1.shape[-1])
    q0 = rearrange(flows["base"], "b w (h a) -> b w h a", h=q1.shape[-2], a=q1.shape[-1])
    qvh = rearrange(flows["pred"], "b w (h a) -> b w h a", h=q1.shape[-2], a=q1.shape[-1])
    qt = qt[..., :-1]  # no gripper
    q1 = q1[..., :-1]  # no gripper
    q0 = q0[..., :-1]  # no gripper
    qvh = qvh[..., :-1]  # no gripper
    t = jnp.expand_dims(flows["time"], axis=-1)

    if True:
        x1 = fk(q1)
        xt = fk(qt)
        x0 = fk(q0)
        xvh = fk(q0 + qvh) - fk(q0)  # task space displacement
        xv = x1 - x0

    pred, target = xvh, xv
    if action_head_mask is None:
        action_head_mask = jnp.ones(pred.shape[0], dtype=bool)

    mask = timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]
    mask = rearrange(mask[..., : pred.shape[-1]], "b w h a -> b w (h a)")
    pred, target = rearrange(pred, "b w h a -> b w (h a)"), rearrange(target, "b w h a -> b w (h a)")

    loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
    return loss, metrics


class AdjFlowHead(FlowMatchingActionHead):
    """Flow matching head with cartesian space adjustments."""

    wf: float = 0.5  # flow loss weight
    wa: float = 0.5  # adjustor loss weight

    def setup(self):
        super().setup()
        from crossformer.model.components.adj.cart import get_fwd_kin_fn, get_jac_fn, make_robot

        robot = make_robot()
        self.adjustor = partial(
            _cart_flow_adjustor,
            fk=get_fwd_kin_fn(robot, pad_gripper=True),
            J_fn=get_jac_fn(robot, pad_gripper=True),
        )

    def do_flow(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        """Execute flow matching."""
        if self.constrain_loss_dims:
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        rng = self.make_rng("dropout")
        base_key, time_key = jax.random.split(rng)
        base = self.base_std * jax.random.normal(base_key, actions_flat.shape)

        time = sample_tau(time_key, shape=(*actions_flat.shape[:2], 1), s=0.99)

        blended = time * actions_flat + (1.0 - time) * base
        target = actions_flat - base

        pred = self(
            transformer_outputs,
            time=time,
            a_t=blended,
            train=train,
        )

        return {
            "pred": pred,
            "target": target,
            "time": time,
            "base": base,
            "blended": blended,
            "action_pad_mask": action_pad_mask,
        }

    def flow_loss(
        self,
        pred,
        target,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ):
        """Compute flow loss."""
        if action_head_mask is None:
            action_head_mask = jnp.ones(pred.shape[0], dtype=bool)

        mask = rearrange(
            (timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]),
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred, target, mask, loss_type="mse")
        return loss, metrics

    def loss(self, *, embeddings, batch, train=True) -> tuple[Array, dict[str, Array]]:
        """Compute combined flow and adjustment loss."""
        timestep_pad_mask = batch["observation"]["timestep_pad_mask"].astype(bool)
        action_pad_mask = batch["action_pad_mask"].astype(bool)
        action_head_mask = batch["action_head_masks"]["single_arm"].astype(bool)

        flows = self.do_flow(embeddings, batch["action"], action_pad_mask=action_pad_mask, train=train)
        use_flow_head = action_head_mask & batch["mask"]["only_adjustment"]
        lf, fmetrics = self.flow_loss(
            flows["pred"],
            flows["target"],
            timestep_pad_mask=timestep_pad_mask,
            action_pad_mask=flows["action_pad_mask"],
            # if only adjustment then zero out flow loss
            action_head_mask=use_flow_head,
            train=train,
        )
        la, ametrics = self.adjustor(
            batch,
            flows,
            timestep_pad_mask=timestep_pad_mask,
            action_pad_mask=flows["action_pad_mask"],
            action_head_mask=action_head_mask,
        )
        l = (lf * self.wf) + (la * self.wa)
        return l, {"flow": fmetrics, "adj": ametrics, "combined": {"lf": lf, "la": la, "total": l}}
