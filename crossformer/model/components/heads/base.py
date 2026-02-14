"""Base action head classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from einops import rearrange
import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.transformer import MAPHead
from crossformer.utils.mytyping import PRNGKey

from .losses import continuous_loss


class ActionHead(ABC):
    """Action prediction modules that take in the transformer token outputs and predict actions.

    Each action head does chunked action prediction: at every timestep, it predicts the next
    `action_horizon` actions into the future from that timestep. Setting `action_horizon=1`
    corresponds to the typical action prediction setup.
    """

    @abstractmethod
    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        argmax: bool = False,
        sample_shape: tuple[int, ...] = (),
        rng: PRNGKey | None = None,
        temperature: float = 1.0,
        train: bool = False,
        embodiment_action_dim: int | None = None,
    ) -> Array:
        """Predict the action for the last timestep in the window.

        Returns shape (*sample_shape, batch_size, action_horizon, action_dim).
        """
        raise NotImplementedError


class ContinuousActionHead(nn.Module, ActionHead):
    """Predicts continuous actions via regression.

    Continuous actions are predicted by tanh squashing the model output to [-max_action, max_action],
    and optimized using a standard regression loss.

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

    def setup(self):
        if self.pool_strategy == "use_map":
            self.map_head = MAPHead()

        num_preds = self.num_preds if self.num_preds else self.action_horizon * self.action_dim
        self.mean_proj = nn.Dense(num_preds)

    def __call__(self, transformer_outputs: dict[str, TokenGroup], train: bool = True) -> jax.Array:
        """Predict actions.

        Returns:
            mean: Predicted actions w/ shape (batch_size, window_size, action_horizon, action_dim)
        """
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

        if len(embeddings.shape) == 3:
            # Implies embeddings is (batch_size, window_size, embedding_size)
            mean = self.mean_proj(embeddings)
            mean = rearrange(mean, "b w (h a) -> b w h a", h=self.action_horizon, a=self.action_dim)
        else:
            # Assumes embeddings is (batch_size, window_size, H, embedding_size)
            assert embeddings.shape[-2] == self.action_horizon
            mean = self.mean_proj(embeddings)

        if self.clip_pred:
            mean = jnp.tanh(mean / self.max_action) * self.max_action

        return mean

    def loss(
        self,
        transformer_outputs: dict[str, TokenGroup],
        actions: ArrayLike,
        timestep_pad_mask: ArrayLike,
        action_pad_mask: ArrayLike,
        action_head_mask: ArrayLike | None = None,
        train: bool = True,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute regression loss.

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
        if self.constrain_loss_dims:
            # Constrain loss to action dimensions and horizon specific to this head
            actions = actions[:, :, : self.action_horizon, : self.action_dim]
            action_pad_mask = action_pad_mask[:, :, : self.action_horizon, : self.action_dim]

        # (batch, window_size, action_horizon, action_dim)
        mean = self(transformer_outputs, train=train)

        if action_head_mask is None:
            action_head_mask = jnp.ones(mean.shape[0], dtype=bool)

        # Combine timestep pad mask with action pad mask and action head mask
        mask = timestep_pad_mask[:, :, None, None] & action_pad_mask & action_head_mask[:, None, None, None]

        loss, metrics = continuous_loss(mean, actions, mask, loss_type=self.loss_type)
        return loss, metrics

    def predict_action(
        self,
        transformer_outputs: dict[str, TokenGroup],
        train: bool = True,
        *args,
        sample_shape: tuple = (),
        **kwargs,
    ) -> jax.Array:
        """Predict actions for the final timestep in the window."""
        mean = self(transformer_outputs, train=train)
        return jnp.broadcast_to(mean, sample_shape + mean.shape)
