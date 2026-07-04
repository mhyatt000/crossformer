from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.tokenizers import LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.xflow_eval import extract_bundled_actions, flatten_obs

pytestmark = pytest.mark.integration


def _spec(cls: type[object], **kwargs: object) -> dict[str, Any]:
    return {"module": cls.__module__, "name": cls.__name__, "args": (), "kwargs": kwargs}


def _example_batch() -> dict[str, Any]:
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


def _model_config(*, max_h: int, max_a: int, max_w: int) -> dict[str, Any]:
    token_dim, transformer_kwargs = common_transformer_sizes("dummy")
    return {
        "model": {
            "observation_tokenizers": {
                "proprio": _spec(LowdimObsTokenizer, obs_keys=("proprio_.*",)),
            },
            "task_tokenizers": {},
            "heads": {
                "action": _spec(
                    XFlowHead,
                    readout_key="readout_action",
                    max_horizon=max_h,
                    max_dofs=max_a,
                    num_query_channels=32,
                    num_heads=2,
                    num_blocks=1,
                    num_self_attend_layers=1,
                    dropout_prob=0.0,
                    flow_steps=3,
                )
            },
            "readouts": {"action": 4},
            "transformer_kwargs": transformer_kwargs,
            "token_embedding_size": token_dim,
            "max_horizon": max_w,
        }
    }


def test_xflow_script_config_forward_smoke() -> None:
    batch = _example_batch()
    obs = flatten_obs(batch["observation"], ("proprio_pose",))
    max_h = batch["act"]["base"].shape[1]
    max_a = batch["act"]["id"].shape[-1]
    max_w = obs["timestep_pad_mask"].shape[1]

    model = CrossFormerModel.from_config(
        _model_config(max_h=max_h, max_a=max_a, max_w=max_w),
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
    assert "readout_action" in outputs
    assert outputs["readout_action"].tokens.shape[:2] == (2, 1)
    assert jnp.all(jnp.isfinite(outputs["readout_action"].tokens))

    actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h)
    pred = model.sample_actions(
        obs,
        batch["task"],
        timestep_pad_mask=obs["timestep_pad_mask"],
        rng=jax.random.PRNGKey(1),
        train=False,
        head_name="action",
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
    )

    assert actions.shape == (2, 1, 4, 5)
    assert pred.shape == actions.shape
    assert pred.dtype == actions.dtype
    assert jnp.all(jnp.isfinite(pred))
