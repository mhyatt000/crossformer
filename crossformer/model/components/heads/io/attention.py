"""Perceiver IO attention primitives (Flax port)."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


def make_view_bias_weight(
    q_view: ArrayLike,
    kv_view: ArrayLike,
    *,
    same_view: float = 1.2,
    other_view: float = 0.8,
    neutral: float = 1.0,
) -> Array:
    """Multiplicative attention weights biasing each query toward its own view.

    View id 0 = NO_VIEW is neutral on either side; a kv token with view >= 1 is an
    image token of that camera view, view 0 is a global/view-independent token. A
    query and kv token are paired only when both have view >= 1.

    Args:
        q_view: [batch, q_len] int view id per query.
        kv_view: [batch, kv_len] int view id per context token.

    Returns:
        [batch, q_len, kv_len] float multiplicative weights.
    """
    q_view = jnp.asarray(q_view)
    kv_view = jnp.asarray(kv_view)

    same = q_view[:, :, None] == kv_view[:, None, :]
    paired = (q_view[:, :, None] > 0) & (kv_view[:, None, :] > 0)

    dtype = jnp.result_type(same_view, other_view, neutral)
    w = jnp.full(same.shape, neutral, dtype=dtype)
    w = jnp.where(paired & same, same_view, w)
    w = jnp.where(paired & ~same, other_view, w)
    return w


def attend(
    q: Array,
    k: Array,
    v: Array,
    dropout_prob: float = 0.0,
    attention_mask: ArrayLike | None = None,
    bias_weight: ArrayLike | None = None,
    deterministic: bool = True,
) -> Array:
    """Multi-head attention.

    Args:
        q: [batch, q_len, heads, head_dim]
        k: [batch, kv_len, heads, head_dim]
        v: [batch, kv_len, heads, v_head_dim]
        dropout_prob: attention dropout rate.
        attention_mask: [batch, q_len, kv_len] bool mask.
        bias_weight: [batch, q_len, kv_len] multiplicative attention weights,
            applied additively in the log domain after the hard mask.
        deterministic: if False, apply dropout.

    Returns:
        [batch, q_len, heads * v_head_dim]
    """
    batch, q_len, num_heads, q_head_dim = q.shape
    _, _, _, v_head_dim = v.shape
    hiddens = num_heads * v_head_dim

    attn = jnp.einsum("bthd,bThd->bhtT", q, k)
    attn *= 1.0 / math.sqrt(q_head_dim)

    large_k = jnp.array(1e4 if attn.dtype == jnp.float16 else 1e30, dtype=attn.dtype)

    if attention_mask is not None:
        attn = jnp.where(attention_mask[:, None, :, :], attn, -large_k)

    if bias_weight is not None:
        bias_weight = jnp.asarray(bias_weight, dtype=attn.dtype)
        bias = jnp.where(bias_weight > 0, jnp.log(bias_weight), -large_k)
        attn = attn + bias[:, None, :, :]

    attn = jax.nn.softmax(attn)

    if dropout_prob > 0 and not deterministic:
        attn = nn.Dropout(rate=dropout_prob, deterministic=False)(attn)

    out = jnp.einsum("bhtT,bThd->bthd", attn, v)
    out = jnp.reshape(out, [batch, q_len, hiddens])

    if attention_mask is not None:
        wipe = jnp.all(attention_mask == 0, axis=2, keepdims=True)
        out = jnp.where(wipe, jnp.zeros_like(out), out)

    return out


def make_cross_attention_mask(query_mask: ArrayLike, kv_mask: ArrayLike) -> Array:
    """Outer-product attention mask from per-sequence masks."""
    mask = jax.vmap(jnp.outer)(query_mask, kv_mask)
    return mask


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention."""

    num_heads: int = 8
    init_scale: float = 1.0
    with_final_bias: bool = True
    final_init_scale_multiplier: float = 1.0
    dropout_prob: float = 0.0
    qk_channels: int | None = None
    v_channels: int | None = None
    output_channels: int | None = None

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        attention_mask: ArrayLike | None = None,
        bias_weight: ArrayLike | None = None,
        deterministic: bool = True,
    ) -> Array:
        qk_channels = self.qk_channels or inputs_q.shape[-1]
        v_channels = self.v_channels or qk_channels
        output_channels = self.output_channels or v_channels

        if qk_channels % self.num_heads != 0:
            raise ValueError(f"qk_channels ({qk_channels}) must be divisible by num_heads ({self.num_heads}).")
        if v_channels % self.num_heads != 0:
            raise ValueError(f"v_channels ({v_channels}) must be divisible by num_heads ({self.num_heads}).")

        qk_head = qk_channels // self.num_heads
        v_head = v_channels // self.num_heads

        kernel_init = nn.initializers.variance_scaling(self.init_scale, "fan_in", "truncated_normal")
        final_init = nn.initializers.variance_scaling(
            self.final_init_scale_multiplier * self.init_scale,
            "fan_in",
            "truncated_normal",
        )

        q = nn.Dense(qk_channels, kernel_init=kernel_init, name="q")(inputs_q)
        k = nn.Dense(qk_channels, kernel_init=kernel_init, name="k")(inputs_kv)
        v = nn.Dense(v_channels, kernel_init=kernel_init, name="v")(inputs_kv)

        batch, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        q = jnp.reshape(q, [batch, q_len, self.num_heads, qk_head])
        k = jnp.reshape(k, [batch, kv_len, self.num_heads, qk_head])
        v = jnp.reshape(v, [batch, kv_len, self.num_heads, v_head])

        result = attend(
            q,
            k,
            v,
            dropout_prob=self.dropout_prob,
            attention_mask=attention_mask,
            bias_weight=bias_weight,
            deterministic=deterministic,
        )
        return nn.Dense(
            output_channels,
            use_bias=self.with_final_bias,
            kernel_init=final_init,
            name="output",
        )(result)


class MLP(nn.Module):
    """Post-attention feed-forward block."""

    widening_factor: int = 4
    dropout_prob: float = 0.0
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, x: Array, deterministic: bool = True) -> Array:
        out_ch = x.shape[-1]
        kernel_init = nn.initializers.variance_scaling(self.init_scale, "fan_in", "truncated_normal")
        x = nn.Dense(self.widening_factor * out_ch, kernel_init=kernel_init, name="up")(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(out_ch, kernel_init=kernel_init, name="down")(x)
        if self.dropout_prob > 0 and not deterministic:
            x = nn.Dropout(rate=self.dropout_prob, deterministic=False)(x)
        return x


class SelfAttention(nn.Module):
    """Pre-norm self-attention + MLP block."""

    num_heads: int = 8
    widening_factor: int = 4
    dropout_prob: float = 0.0
    dropout_attn_prob: float = 0.0
    att_init_scale: float = 1.0
    dense_init_scale: float = 1.0
    qk_channels: int | None = None
    v_channels: int | None = None

    @nn.compact
    def __call__(self, inputs: Array, attention_mask: ArrayLike | None = None, deterministic: bool = True) -> Array:
        x = inputs
        qkv = nn.LayerNorm(name="ln_attn")(inputs)
        attn = Attention(
            num_heads=self.num_heads,
            init_scale=self.att_init_scale,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            dropout_prob=self.dropout_attn_prob,
            name="attention",
        )(qkv, qkv, attention_mask=attention_mask, deterministic=deterministic)

        if self.dropout_prob > 0 and not deterministic:
            attn = nn.Dropout(rate=self.dropout_prob, deterministic=False)(attn)
        x = x + attn

        x = x + MLP(
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            init_scale=self.dense_init_scale,
            name="mlp",
        )(nn.LayerNorm(name="ln_mlp")(x), deterministic=deterministic)
        return x


class CrossAttention(nn.Module):
    """Pre-norm cross-attention + MLP block."""

    num_heads: int = 8
    widening_factor: int = 1
    dropout_prob: float = 0.0
    dropout_attn_prob: float = 0.0
    att_init_scale: float = 1.0
    dense_init_scale: float = 1.0
    shape_for_attn: str = "kv"
    use_query_residual: bool = True
    qk_channels: int | None = None
    v_channels: int | None = None

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        attention_mask: ArrayLike | None = None,
        bias_weight: ArrayLike | None = None,
        deterministic: bool = True,
    ) -> Array:
        output_channels = inputs_q.shape[-1]

        if self.shape_for_attn == "q":
            qk_channels = inputs_q.shape[-1]
        elif self.shape_for_attn == "kv":
            qk_channels = inputs_kv.shape[-1]
        else:
            raise ValueError(f"Unknown shape_for_attn: {self.shape_for_attn}")

        if self.qk_channels is not None:
            qk_channels = self.qk_channels

        attn = Attention(
            num_heads=self.num_heads,
            init_scale=self.att_init_scale,
            dropout_prob=self.dropout_attn_prob,
            qk_channels=qk_channels,
            v_channels=self.v_channels,
            output_channels=output_channels,
            name="attention",
        )(
            nn.LayerNorm(name="ln_q")(inputs_q),
            nn.LayerNorm(name="ln_kv")(inputs_kv),
            attention_mask=attention_mask,
            bias_weight=bias_weight,
            deterministic=deterministic,
        )

        if self.dropout_prob > 0 and not deterministic:
            attn = nn.Dropout(rate=self.dropout_prob, deterministic=False)(attn)

        x = inputs_q + attn if self.use_query_residual else attn

        x = x + MLP(
            widening_factor=self.widening_factor,
            dropout_prob=self.dropout_prob,
            init_scale=self.dense_init_scale,
            name="mlp",
        )(nn.LayerNorm(name="ln_mlp")(x), deterministic=deterministic)
        return x
