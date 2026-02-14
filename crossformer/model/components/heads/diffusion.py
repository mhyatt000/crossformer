"""Diffusion-based action head."""

from __future__ import annotations

from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.diffusion import (
    cosine_beta_schedule,
    create_diffusion_model,
)
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.mytyping import PRNGKey

from .base import ActionHead
from .losses import continuous_loss


class DiffusionActionHead(nn.Module, ActionHead):
    """Predicts actions using a diffusion process.

    Only a single pass through the transformer obtains an action embedding at each timestep.
    The action is then predicted using a diffusion process conditioned on this embedding.
    The diffusion model uses an MLP with residual connections.

    May create an embedding by either mean-pooling across tokens (use_map=False) or using
    multi-head attention pooling (use_map=True). MAP is recommended for observation tokens.
    """

    readout_key: str
    pool_strategy: str = "mean"
    action_horizon: int = 1
    action_dim: int = 7
    clip_pred: bool = True
    max_action: float = 5.0
    loss_type: str = "mse"
    num_preds: int = 0
    loss_weight: float = 1.0
    constrain_loss_dims: bool = True

    # Diffusion-specific config with sane defaults
    time_dim: int = 32
    num_blocks: int = 3
    dropout_rate: float = 0.1
    hidden_dim: int = 256
    use_layer_norm: bool = True
    diffusion_steps: int = 20

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        # Create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            self.action_dim * self.action_horizon,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate,
            hidden_dim=self.hidden_dim,
            use_layer_norm=self.use_layer_norm,
        )

        # Create beta schedule
        self.betas = jnp.array(cosine_beta_schedule(self.diffusion_steps))
        self.alphas = 1 - self.betas
        self.alpha_hats = jnp.array([jnp.prod(self.alphas[: i + 1]) for i in range(self.diffusion_steps)])

    def __call__(
        self,
        transformer_outputs: dict[str, TokenGroup],
        time: ArrayLike | None = None,
        noisy_actions: ArrayLike | None = None,
        train: bool = True,
    ) -> jax.Array:
        """Perform a single forward pass through the diffusion model."""
        token_group = transformer_outputs[self.readout_key]
        assert token_group.tokens.ndim == 4, (
            f"Expected token_group.tokens to have shape (batch_size, window_size, num_tokens, embedding_size), "
            f"but got shape {token_group.tokens.shape}"
        )

        if self.pool_strategy == "use_map":  # Multi-head attention pooling
            embeddings = self.map_head(token_group, train=train)[:, :, 0]
        elif self.pool_strategy == "mean":  # mean pooling
            embeddings = token_group.tokens.mean(axis=-2)
        elif self.pool_strategy == "pass":
            embeddings = token_group.tokens
        else:
            raise ValueError(f"{self.pool_strategy} not implemented!")

        # Now, embeddings is (batch_size, window_size, embedding_size)

        # time and noisy_actions are None during initialization, so use dummy arrays
        if (time is None or noisy_actions is None) and not self.is_initializing():
            raise ValueError("Must provide time and noisy_actions when calling diffusion action head")
        elif self.is_initializing():
            time = jnp.zeros((*embeddings.shape[:2], 1), dtype=jnp.float32)
            noisy_actions = jnp.zeros(
                (*embeddings.shape[:2], self.action_dim * self.action_horizon),
                dtype=jnp.float32,
            )

        pred_eps = self.diffusion_model(embeddings, noisy_actions, time, train=train)
        return pred_eps

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute diffusion loss.

        Args:
            transformer_outputs: must contain self.readout_key with shape
                (batch_size, window_size, num_tokens, embedding_size)
            actions: shape (batch_size, window_size, action_horizon, action_dim)
            timestep_pad_mask: boolean array (batch, window_size) True if timestep is not padding
            action_pad_mask: boolean array (same shape as actions) True if action dim is not padding
            action_head_mask: boolean array (batch,) True if this action space is used

        Returns:
            loss: float
            metrics: dict
        """
        batch_size, window_size = timestep_pad_mask.shape

        if self.constrain_loss_dims:
            # Constrain loss to action dimensions and horizon specific to this head
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        # Reshape and clip actions
        actions_flat = rearrange(actions, "b w h a -> b w (h a)")
        actions_flat = jnp.clip(actions_flat, -self.max_action, self.max_action)

        # Sample noise and timesteps
        rng = self.make_rng("dropout")
        time_key, noise_key = jax.random.split(rng)
        time = jax.random.randint(time_key, (batch_size, window_size, 1), 0, self.diffusion_steps)
        noise = jax.random.normal(noise_key, actions_flat.shape)

        alpha_hat = self.alpha_hats[time]
        alpha_1 = jnp.sqrt(alpha_hat)
        alpha_2 = jnp.sqrt(1 - alpha_hat)
        noisy_actions = alpha_1 * actions_flat + alpha_2 * noise

        pred_eps = self(transformer_outputs, train=train, time=time, noisy_actions=noisy_actions)

        if action_head_mask is None:
            action_head_mask = jnp.ones(pred_eps.shape[0], dtype=bool)

        # Combine the timestep pad mask with the action pad mask and the action head mask
        mask = rearrange(
            (
                timestep_pad_mask[:, :, None, None]  # dimension reduced because noise is b w (h a)
                & action_pad_mask
                & action_head_mask[:, None, None, None]
            ),
            "b w h a -> b w (h a)",
        )

        loss, metrics = continuous_loss(pred_eps, noise, mask, loss_type=self.loss_type)
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        rng: PRNGKey,
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Predict actions by sampling from the diffusion process."""
        module, variables = self.unbind()

        def scan_fn(carry, time):
            current_x, rng = carry
            input_time = jnp.broadcast_to(time, (*current_x.shape[:-1], 1))

            eps_pred = module.apply(variables, transformer_outputs, input_time, current_x, train=train)

            alpha_1 = 1 / jnp.sqrt(self.alphas[time])
            alpha_2 = (1 - self.alphas[time]) / (jnp.sqrt(1 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(key, shape=current_x.shape)
            current_x = current_x + (time > 0) * (jnp.sqrt(self.betas[time]) * z)

            current_x = jnp.clip(current_x, -self.max_action, self.max_action)

            return (current_x, rng), ()

        def sample_actions(rng):
            rng, key = jax.random.split(rng)
            batch_size, window_size = transformer_outputs[self.readout_key].tokens.shape[:2]

            (actions_flat, _), () = jax.lax.scan(
                scan_fn,
                (
                    jax.random.normal(
                        key,
                        (
                            batch_size,
                            window_size,
                            self.action_horizon * self.action_dim,
                        ),
                    ),
                    rng,
                ),
                jnp.arange(self.diffusion_steps - 1, -1, -1),
            )

            actions = rearrange(
                actions_flat,
                "b w (h a) -> b w h a",
                h=self.action_horizon,
                a=self.action_dim,
            )
            return actions

        n_samples = int(np.prod(sample_shape)) if sample_shape else 1
        actions = jax.vmap(sample_actions)(jax.random.split(rng, n_samples))
        actions = actions.reshape(sample_shape + actions.shape[1:])
        return actions
