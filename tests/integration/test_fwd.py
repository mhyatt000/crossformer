from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.crossformer_model import CrossFormerModel
from scripts.train.xflow import Config, extract_bundled_actions, make_model_config, normalize_obs, resolve_obs_keys

pytestmark = pytest.mark.integration


def _example_batch():
    b, w, h, a = 2, 1, 4, 5
    obs = {
        "proprio_joint": jnp.arange(b * w * 7, dtype=jnp.float32).reshape(b, w, 7) / 10.0,
        "proprio_pose": jnp.arange(b * w * 2 * 3, dtype=jnp.float32).reshape(b, w, 2, 3) / 20.0,
        "timestep_pad_mask": jnp.ones((b, w), dtype=jnp.bool_),
        "pad_mask_dict": {
            "proprio_joint": jnp.ones((b, w), dtype=jnp.bool_),
            "proprio_pose": jnp.ones((b, w), dtype=jnp.bool_),
        },
    }
    act = {
        "base": jnp.arange(b * h * a, dtype=jnp.float32).reshape(b, h, a) / 50.0,
        "id": jnp.array(
            [
                [1, 2, 3, 4, 0],
                [5, 6, 7, 0, 0],
            ],
            dtype=jnp.int32,
        ),
    }
    return {
        "observation": obs,
        "task": {"pad_mask_dict": {}},
        "act": act,
    }


def test_xflow_script_config_forward_smoke():
    batch = _example_batch()
    obs_keys = resolve_obs_keys(batch["observation"], ("proprio_.*",))
    obs = normalize_obs(batch["observation"], obs_keys)
    max_h = batch["act"]["base"].shape[1]
    max_a = batch["act"]["id"].shape[-1]
    max_w = obs["timestep_pad_mask"].shape[1]

    cfg = Config(
        transformer_size="dummy",
        obs_keys=("proprio_.*",),
        use_vision=False,
        use_guidance=False,
        head_channels=32,
        head_depth=1,
        head_heads=2,
        flow_steps=3,
    )
    config = make_model_config(cfg, max_h=max_h, max_a=max_a, max_w=max_w)

    model = CrossFormerModel.from_config(
        config,
        {"observation": obs, "task": batch["task"]},
        text_processor=None,
        verbose=False,
        rng=jax.random.PRNGKey(0),
        dataset_statistics=None,
    )

    outputs = model.run_transformer(
        obs,
        batch["task"],
        obs["timestep_pad_mask"],
        train=False,
    )
    assert "readout_xflow" in outputs
    assert outputs["readout_xflow"].tokens.shape[:2] == (2, 1)
    assert jnp.all(jnp.isfinite(outputs["readout_xflow"].tokens))

    actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h)
    pred = model.sample_actions(
        obs,
        batch["task"],
        timestep_pad_mask=obs["timestep_pad_mask"],
        rng=jax.random.PRNGKey(1),
        train=False,
        head_name="xflow",
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
    )

    assert actions.shape == (2, 1, 4, 5)
    assert pred.shape == actions.shape
    assert pred.dtype == actions.dtype
    assert jnp.all(jnp.isfinite(pred))
