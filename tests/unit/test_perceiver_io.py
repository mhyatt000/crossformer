"""Smoke tests for Perceiver IO Flax modules."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.heads.io import (
    Attention,
    BasicDecoder,
    CrossAttention,
    FourierPositionEncoding,
    MLP,
    Perceiver,
    PerceiverEncoder,
    ProjectionDecoder,
    SelfAttention,
    TrainablePositionEncoding,
)

pytestmark = pytest.mark.nn

RNG = jax.random.PRNGKey(0)
BATCH, SEQ, DIM = 2, 10, 32


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture
def inputs():
    return jnp.ones((BATCH, SEQ, DIM))


# ---- position encodings ----


def test_trainable_position_encoding(rng):
    enc = TrainablePositionEncoding(index_dim=8, num_channels=16)
    params = enc.init(rng, batch_size=BATCH)
    out = enc.apply(params, batch_size=BATCH)
    assert out.shape == (BATCH, 8, 16)


def test_fourier_position_encoding(rng):
    enc = FourierPositionEncoding(index_dims=(4, 4), num_bands=3, concat_pos=True)
    params = enc.init(rng, batch_size=BATCH)
    out = enc.apply(params, batch_size=BATCH)
    assert out.shape[0] == BATCH
    assert out.shape[1] == 16  # 4*4
    assert out.ndim == 3


# ---- attention primitives ----


def test_attention_self(rng, inputs):
    attn = Attention(num_heads=4)
    params = attn.init(rng, inputs, inputs)
    out = attn.apply(params, inputs, inputs)
    assert out.shape == inputs.shape


def test_attention_cross(rng, inputs):
    kv = jnp.ones((BATCH, 6, 16))
    attn = Attention(num_heads=4, qk_channels=16, v_channels=16, output_channels=DIM)
    params = attn.init(rng, inputs, kv)
    out = attn.apply(params, inputs, kv)
    assert out.shape == inputs.shape


def test_mlp(rng, inputs):
    mlp = MLP(widening_factor=2)
    params = mlp.init(rng, inputs)
    out = mlp.apply(params, inputs)
    assert out.shape == inputs.shape


def test_self_attention_block(rng, inputs):
    block = SelfAttention(num_heads=4, widening_factor=2)
    params = block.init(rng, inputs)
    out = block.apply(params, inputs)
    assert out.shape == inputs.shape


def test_cross_attention_block(rng, inputs):
    kv = jnp.ones((BATCH, 6, 16))
    block = CrossAttention(num_heads=4)
    params = block.init(rng, inputs, kv)
    out = block.apply(params, inputs, kv)
    assert out.shape == inputs.shape


# ---- encoder ----


def test_encoder_init_and_forward(rng, inputs):
    enc = PerceiverEncoder(
        num_self_attends_per_block=2,
        num_blocks=1,
        z_index_dim=8,
        num_z_channels=DIM,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
    )
    # latents() needs setup, so init with dummy z first
    z_init = jnp.zeros((BATCH, 8, DIM))
    params = enc.init(rng, inputs, z_init, deterministic=True)
    z = enc.apply(params, inputs, z_init, deterministic=True)
    assert z.shape == (BATCH, 8, DIM)


def test_encoder_with_input_mask(rng, inputs):
    enc = PerceiverEncoder(
        num_self_attends_per_block=1,
        num_blocks=1,
        z_index_dim=4,
        num_z_channels=DIM,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
    )
    mask = jnp.ones((BATCH, SEQ), dtype=jnp.int32)
    mask = mask.at[:, -3:].set(0)  # mask last 3 positions
    z_init = jnp.zeros((BATCH, 4, DIM))
    params = enc.init(
        rng,
        inputs,
        z_init,
        deterministic=True,
        input_mask=mask,
    )
    z = enc.apply(
        params,
        inputs,
        z_init,
        deterministic=True,
        input_mask=mask,
    )
    assert z.shape == (BATCH, 4, DIM)


# ---- decoders ----


def test_projection_decoder(rng):
    dec = ProjectionDecoder(num_classes=10)
    z = jnp.ones((BATCH, 8, DIM))
    params = dec.init(rng, None, z)
    out = dec.apply(params, None, z)
    assert out.shape == (BATCH, 10)


def test_basic_decoder(rng):
    """Test BasicDecoder via full Perceiver (decoder_query needs module scope)."""
    enc = PerceiverEncoder(
        num_self_attends_per_block=1,
        num_blocks=1,
        z_index_dim=8,
        num_z_channels=DIM,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
    )
    dec = BasicDecoder(
        output_num_channels=16,
        output_index_dims=(4,),
        position_encoding_type="trainable",
        position_encoding_kwargs={
            "trainable_position_encoding_kwargs": {"num_channels": DIM},
        },
        num_heads=1,
    )
    model = Perceiver(encoder=enc, decoder=dec)
    x = jnp.ones((BATCH, SEQ, DIM))
    params = model.init(rng, x, deterministic=True)
    out = model.apply(params, x, deterministic=True)
    assert out.shape == (BATCH, 4, 16)


# ---- full perceiver ----


def test_perceiver_forward(rng, inputs):
    enc = PerceiverEncoder(
        num_self_attends_per_block=2,
        num_blocks=1,
        z_index_dim=8,
        num_z_channels=DIM,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
    )
    dec = BasicDecoder(
        output_num_channels=16,
        output_index_dims=(4,),
        position_encoding_type="trainable",
        position_encoding_kwargs={
            "trainable_position_encoding_kwargs": {"num_channels": DIM},
        },
        num_heads=1,
    )
    model = Perceiver(encoder=enc, decoder=dec)
    params = model.init(rng, inputs, deterministic=True)
    out = model.apply(params, inputs, deterministic=True)
    assert out.shape == (BATCH, 4, 16)


def test_perceiver_output_dtype(rng, inputs):
    enc = PerceiverEncoder(
        num_self_attends_per_block=1,
        num_blocks=1,
        z_index_dim=4,
        num_z_channels=DIM,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
    )
    dec = ProjectionDecoder(num_classes=5)
    model = Perceiver(encoder=enc, decoder=dec)
    params = model.init(rng, inputs, deterministic=True)
    out = model.apply(params, inputs, deterministic=True)
    assert out.dtype == jnp.float32
    assert out.shape == (BATCH, 5)


def test_perceiver_determinism(rng, inputs):
    """Same params + input should give identical outputs."""
    enc = PerceiverEncoder(
        num_self_attends_per_block=1,
        num_blocks=1,
        z_index_dim=4,
        num_z_channels=DIM,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
    )
    dec = BasicDecoder(
        output_num_channels=8,
        output_index_dims=(3,),
        position_encoding_type="trainable",
        position_encoding_kwargs={
            "trainable_position_encoding_kwargs": {"num_channels": DIM},
        },
        num_heads=1,
    )
    model = Perceiver(encoder=enc, decoder=dec)
    params = model.init(rng, inputs, deterministic=True)
    out1 = model.apply(params, inputs, deterministic=True)
    out2 = model.apply(params, inputs, deterministic=True)
    assert jnp.allclose(out1, out2)
