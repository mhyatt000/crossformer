"""Position encodings for Perceiver IO (Flax port)."""

from __future__ import annotations

import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def generate_fourier_features(
    pos,
    num_bands,
    max_resolution=(224, 224),
    concat_pos=True,
    sine_only=False,
):
    """Fourier frequency position encoding with linear spacing.

    Args:
        pos: positions [n, d].
        num_bands: number of frequency bands (K).
        max_resolution: max resolution per dim.
        concat_pos: append raw positions to output.
        sine_only: use sin only (not sin+cos).

    Returns:
        [n, n_channels] position features.
    """
    min_freq = 1.0
    freq_bands = jnp.stack(
        [jnp.linspace(min_freq, res / 2, num=num_bands, endpoint=True) for res in max_resolution],
        axis=0,
    )

    per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
    per_pos_features = jnp.reshape(per_pos_features, [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        per_pos_features = jnp.sin(jnp.pi * per_pos_features)
    else:
        per_pos_features = jnp.concatenate(
            [
                jnp.sin(jnp.pi * per_pos_features),
                jnp.cos(jnp.pi * per_pos_features),
            ],
            axis=-1,
        )

    if concat_pos:
        per_pos_features = jnp.concatenate([pos, per_pos_features], axis=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """Generate position indices for N-D input.

    Returns:
        [*index_dims, N] position array.
    """
    dim_ranges = [jnp.linspace(output_range[0], output_range[1], num=n, endpoint=True) for n in index_dims]
    grid = jnp.meshgrid(*dim_ranges, indexing="ij")
    return jnp.stack(grid, axis=-1)


class TrainablePositionEncoding(nn.Module):
    """Trainable position encoding."""

    index_dim: int
    num_channels: int = 128
    init_scale: float = 0.02

    @nn.compact
    def __call__(self, batch_size, pos=None):
        del pos
        pos_embs = self.param(
            "pos_embs",
            nn.initializers.truncated_normal(stddev=self.init_scale),
            (self.index_dim, self.num_channels),
        )
        if batch_size is not None:
            pos_embs = jnp.broadcast_to(pos_embs[None], (batch_size, *pos_embs.shape))
        return pos_embs


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """Build or validate spatial positions [batch, prod(index_dims), d]."""
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = jnp.broadcast_to(pos[None], (batch_size, *pos.shape))
        pos = jnp.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        assert pos.shape[-1] == len(index_dims)
    return pos


class FourierPositionEncoding(nn.Module):
    """Fourier (sinusoidal) position encoding."""

    index_dims: tuple[int, ...]
    num_bands: int
    concat_pos: bool = True
    max_resolution: tuple[int, ...] | None = None
    sine_only: bool = False

    @nn.compact
    def __call__(self, batch_size, pos=None):
        max_resolution = self.max_resolution or self.index_dims
        pos = _check_or_build_spatial_positions(pos, self.index_dims, batch_size)
        build_ff = functools.partial(
            generate_fourier_features,
            num_bands=self.num_bands,
            max_resolution=max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only,
        )
        return jax.vmap(build_ff, 0, 0)(pos)


class PositionEncodingProjector(nn.Module):
    """Projects position encoding to a target size."""

    output_size: int
    base_position_encoding: nn.Module

    @nn.compact
    def __call__(self, batch_size, pos=None):
        base_pos = self.base_position_encoding(batch_size, pos)
        return nn.Dense(self.output_size)(base_pos)


def build_position_encoding(
    position_encoding_type,
    index_dims,
    project_pos_dim=-1,
    trainable_position_encoding_kwargs=None,
    fourier_position_encoding_kwargs=None,
    name=None,
):
    """Factory for position encodings."""
    if position_encoding_type == "trainable":
        assert trainable_position_encoding_kwargs is not None
        enc = TrainablePositionEncoding(
            index_dim=int(np.prod(index_dims)),
            name=name,
            **trainable_position_encoding_kwargs,
        )
    elif position_encoding_type == "fourier":
        assert fourier_position_encoding_kwargs is not None
        enc = FourierPositionEncoding(
            index_dims=index_dims,
            name=name,
            **fourier_position_encoding_kwargs,
        )
    else:
        raise ValueError(f"Unknown position encoding: {position_encoding_type}")

    if project_pos_dim > 0:
        enc = PositionEncodingProjector(
            output_size=project_pos_dim,
            base_position_encoding=enc,
        )
    return enc
