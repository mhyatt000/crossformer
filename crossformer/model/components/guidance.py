"""Guidance token encoders for conditioning flow-matching heads."""

from __future__ import annotations

import flax.linen as nn
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike

from crossformer.model.components.heads.io.attention import CrossAttention, SelfAttention


class TokenGuidance(nn.Module):
    """Project arbitrary input tokens and optionally compress into guidance tokens.

    Two modes controlled by `compress`:
        False (default): Linear project each input → (B, S, E) tokens directly.
        True: Project then compress via learned latent cross-attention → (B, G, E).

    Input: tokens (B, S, D) — any per-step signal (Cartesian pose, actions, etc.).
    Output: (B, G, E) where G = S (direct) or num_latents (compressed).
    """

    embed_dim: int = 256
    compress: bool = False
    num_latents: int = 4
    num_heads: int = 4
    num_compress_layers: int = 1

    @nn.compact
    def __call__(self, tokens: ArrayLike, deterministic: bool = True) -> Array:
        """Encode input tokens to guidance tokens.

        Args:
            tokens: (B, S, D) arbitrary input signal.

        Returns:
            (B, G, E) guidance tokens.
        """
        E = self.embed_dim
        tokens = nn.Dense(E, name="proj")(tokens)  # (B, S, E)
        tokens = nn.LayerNorm(name="ln")(tokens)

        if not self.compress:
            return tokens

        B = tokens.shape[0]
        latents = self.param(
            "latents",
            nn.initializers.xavier_uniform(),
            (1, self.num_latents, E),
        )
        latents = jnp.tile(latents, (B, 1, 1))

        for i in range(self.num_compress_layers):
            latents = CrossAttention(
                num_heads=self.num_heads,
                use_query_residual=True,
                shape_for_attn="kv",
                name=f"compress_xattn_{i}",
            )(latents, tokens, deterministic=deterministic)
            latents = SelfAttention(
                num_heads=self.num_heads,
                name=f"compress_sattn_{i}",
            )(latents, deterministic=deterministic)

        return latents  # (B, G, E)
