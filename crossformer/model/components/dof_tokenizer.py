from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from crossformer.embody import MASK_ID
from crossformer.model.components.base import TokenGroup


class DofTokenizer(nn.Module):
    """Tokenize proprio scalars via a DOF-id-indexed embedding table.

    Consumes:
        observations["proprio"]: (B, T, D) float — z-normalized scalar per slot.
        observations["dof_ids"]: (B, D)   int   — DOF vocab id per slot. MASK_ID = pad.

    Produces a TokenGroup of shape (B, T, D, E), with mask[b, t, d] = False where
    dof_ids[b, d] == MASK_ID so attention skips padded slots.

    The `dof_embed` arg lets the parent module share the same nn.Embed instance
    with the action head's query builder, so proprio input and action output
    land on the same row per DOF.
    """

    embedding_size: int = 512
    dof_vocab_size: int = 32
    p_mask: float = 0.0
    dof_embed: nn.Module | None = None

    @nn.compact
    def __call__(self, observations, tasks=None, train: bool = True):
        x = observations["proprio"]
        dof_ids = observations["dof_ids"]
        E = self.embedding_size

        # Shuffle the D axis per-sample during training so the model learns
        # from DOF id, not slot position. Same permutation across time.
        if train:
            rng = self.make_rng("dropout")
            scores = jax.random.uniform(rng, dof_ids.shape)
            perm = jnp.argsort(scores, axis=-1)
            x = jnp.take_along_axis(x, perm[:, None, :], axis=-1)
            dof_ids = jnp.take_along_axis(dof_ids, perm, axis=-1)

        embed = self.dof_embed or nn.Embed(self.dof_vocab_size, E, name="dof_embed")
        feat_emb = embed(dof_ids)

        # Per-DOF affine modulation: gamma scales the shared value projection per DOF,
        # feat_emb shifts it. Without gamma, every DOF's value would modulate along the
        # same shared direction. feat_emb stays compatible with the head's dof_embed.
        # TODO: init gamma to ones so it starts as a pass-through:
        #   nn.Embed(..., embedding_init=nn.initializers.ones)(dof_ids)
        gamma = nn.Embed(self.dof_vocab_size, E, name="dof_gamma")(dof_ids)
        proj = nn.Dense(E, name="proj")(x[..., None])
        tokens = gamma[:, None] * proj + feat_emb[:, None]

        valid = dof_ids != MASK_ID
        valid = jnp.broadcast_to(valid[:, None, :], x.shape)

        if train and self.p_mask > 0:
            rng = self.make_rng("dropout")
            keep = jax.random.bernoulli(rng, 1 - self.p_mask, valid.shape)
            valid = valid & keep

        return TokenGroup(tokens, valid)
