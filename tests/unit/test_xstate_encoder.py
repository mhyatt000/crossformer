from __future__ import annotations

import jax
import jax.numpy as jnp

from crossformer.cn.model_factory import ModelFactory
from crossformer.model.components.xstate import XStateEncoder


def _obs(*, mask: jax.Array | None = None) -> dict:
    bsz, win, slots = 2, 3, 4
    base = jnp.arange(bsz * win * slots, dtype=jnp.float32).reshape(bsz, win, slots) / 10.0
    if mask is None:
        mask = jnp.array(
            [
                [True, True, True, False],
                [True, False, False, False],
            ],
            dtype=bool,
        )
    return {
        "state": {
            "base": base,
            "id": jnp.array(
                [
                    [1, 2, 3, 0],
                    [4, 0, 0, 0],
                ],
                dtype=jnp.int32,
            ),
            "view": jnp.array(
                [
                    [0, 1, 2, 0],
                    [0, 0, 0, 0],
                ],
                dtype=jnp.int32,
            ),
        },
        "mask": {"states": mask},
    }


def test_xstate_encoder_outputs_token_group() -> None:
    enc = XStateEncoder(num_latents=3, num_channels=32, num_heads=4, num_blocks=1)
    obs = _obs()

    params = enc.init(jax.random.PRNGKey(0), obs, train=False)
    out = enc.apply(params, obs, train=False)

    assert out.tokens.shape == (2, 3, 3, 32)
    assert out.mask.shape == (2, 3, 3)
    assert out.view is not None
    assert out.view.shape == (2, 3, 3)
    assert jnp.all(out.mask)
    assert jnp.all(out.view == 0)
    assert jnp.all(jnp.isfinite(out.tokens))


def test_xstate_encoder_handles_all_masked_states() -> None:
    enc = XStateEncoder(num_latents=2, num_channels=16, num_heads=4, num_blocks=1)
    obs = _obs(mask=jnp.zeros((2, 4), dtype=bool))

    params = enc.init(jax.random.PRNGKey(1), obs, train=False)
    out = enc.apply(params, obs, train=False)

    assert out.tokens.shape == (2, 3, 2, 16)
    assert jnp.all(out.mask)
    assert jnp.all(jnp.isfinite(out.tokens))


def test_xstate_encoder_uses_state_mask() -> None:
    enc = XStateEncoder(num_latents=2, num_channels=16, num_heads=4, num_blocks=1)
    obs = _obs()
    params = enc.init(jax.random.PRNGKey(2), obs, train=False)

    all_mask = obs | {"mask": {"states": jnp.ones((2, 4), dtype=bool)}}
    none_mask = obs | {"mask": {"states": jnp.zeros((2, 4), dtype=bool)}}
    all_out = enc.apply(params, all_mask, train=False).tokens
    none_out = enc.apply(params, none_mask, train=False).tokens

    assert not jnp.allclose(all_out, none_out)


def test_xstate_encoder_latent_drop_masks_whole_samples() -> None:
    enc = XStateEncoder(
        num_latents=2,
        num_channels=16,
        num_heads=4,
        num_blocks=1,
        input_drop_prob=0.0,
        latent_drop_prob=1.0,
    )
    obs = _obs()

    params = enc.init({"params": jax.random.PRNGKey(3), "dropout": jax.random.PRNGKey(4)}, obs, train=True)
    out = enc.apply(params, obs, train=True, rngs={"dropout": jax.random.PRNGKey(5)})

    assert not jnp.any(out.mask)
    assert jnp.all(jnp.isfinite(out.tokens))


def test_model_factory_wires_state_tokenizer() -> None:
    cfg = ModelFactory(image_keys=(), proprio_keys=()).create()["model"]
    state = cfg["observation_tokenizers"]["state"]

    assert state["name"] == "XStateEncoder"
    assert state["kwargs"]["num_latents"] == 8
    assert state["kwargs"]["input_drop_prob"] == 0.25
    assert state["kwargs"]["latent_drop_prob"] == 0.25
