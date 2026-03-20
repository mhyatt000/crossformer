from __future__ import annotations

import jax.numpy as jnp
import pytest

from crossformer.embody import MASK_ID
from crossformer.model.components.heads.dof import CHUNK_PAD, EMBODIMENTS, SLOT_PAD
from scripts.train.xflow import (
    Config,
    HEAD_TO_EMBODIMENT,
    make_model_config,
    normalize_obs,
    prepare_head_inputs,
    resolve_heads,
    resolve_obs_keys,
)


def test_resolve_heads_maps_names_and_counts_dofs():
    info = resolve_heads(("single", "k3ds"))

    assert info["single"]["embodiment"] == HEAD_TO_EMBODIMENT["single"]
    assert info["single"]["n_dofs"] == len(EMBODIMENTS["xarm_gripper"])
    assert info["k3ds"]["n_dofs"] == len(EMBODIMENTS["k3ds"])


def test_resolve_heads_rejects_unknown_head():
    with pytest.raises(ValueError, match="No embodiment mapping for head 'bad'"):
        resolve_heads(("bad",))


def test_make_model_config_wires_xflow_bounds():
    cfg = Config(transformer_size="dummy", obs_keys=("foo", "bar"))

    model_cfg = make_model_config(cfg, max_h=6, max_a=9, max_w=11)["model"]
    head_cfg = model_cfg["heads"]["xflow"]
    head_kwargs = head_cfg["kwargs"]

    assert model_cfg["max_horizon"] == 11
    assert model_cfg["readouts"] == {"xflow": 4}
    assert head_cfg["name"] == "XFlowHead"
    assert head_kwargs["max_horizon"] == 6
    assert head_kwargs["max_dofs"] == 9
    assert head_kwargs["readout_key"] == "readout_xflow"


def test_normalize_obs_adds_channel_and_flattens():
    obs = {
        "scalar": jnp.ones((2, 3)),
        "pose": jnp.ones((2, 3, 2, 4)),
        "already_seq": jnp.ones((2, 3, 5)),
    }

    out = normalize_obs(obs, ("scalar", "pose", "already_seq"))

    assert out["scalar"].shape == (2, 3, 1)
    assert out["pose"].shape == (2, 3, 8)
    assert out["already_seq"].shape == (2, 3, 5)


def test_resolve_obs_keys_preserves_pattern_order_and_deduplicates():
    obs = {
        "joint_pos": None,
        "joint_vel": None,
        "time": None,
        "timestep": None,
    }

    keys = resolve_obs_keys(obs, ("joint_.*", "time", "joint_pos"))

    assert keys == ("joint_pos", "joint_vel", "time")


def test_resolve_obs_keys_requires_match():
    with pytest.raises(ValueError, match="No observation keys matched"):
        resolve_obs_keys({"foo": None}, ("bar",))


def test_prepare_head_inputs_pads_and_flattens_multidim_actions():
    batch = {
        "action": {
            "k3ds": jnp.arange(2 * 3 * 21 * 4, dtype=jnp.float32).reshape(2, 3, 21, 4),
        },
        "embodiment": {
            "k3ds": jnp.array([[1], [0]], dtype=jnp.bool_),
        },
    }

    actions, dof_ids, chunk_steps, slot_pos, emb_mask = prepare_head_inputs(
        batch,
        "k3ds",
        max_h=5,
        max_a=84,
        embodiment_name="k3ds",
    )

    assert actions.shape == (2, 1, 5, 84)
    assert jnp.array_equal(actions[:, 0, :3], batch["action"]["k3ds"].reshape(2, 3, 84))
    assert jnp.all(actions[:, :, 3:, :] == 0)
    assert dof_ids.shape == (2, 84)
    assert chunk_steps.shape == (2, 5)
    assert slot_pos.shape == (2, 84)
    assert emb_mask.shape == (2,)


def test_prepare_head_inputs_fills_padding_metadata_and_default_mask():
    batch = {
        "action": {
            "single": jnp.ones((2, 4, 3), dtype=jnp.float32),
        },
        "embodiment": {},
    }

    actions, dof_ids, chunk_steps, slot_pos, emb_mask = prepare_head_inputs(
        batch,
        "single",
        max_h=6,
        max_a=10,
        embodiment_name="xarm_gripper",
    )

    assert actions.shape == (2, 1, 6, 10)
    assert jnp.all(actions[:, :, 4:, :] == 0)
    assert jnp.all(dof_ids[:, len(EMBODIMENTS["xarm_gripper"]) :] == MASK_ID)
    assert jnp.all(chunk_steps[:, 4:] == CHUNK_PAD)
    assert jnp.all(slot_pos[:, len(EMBODIMENTS["xarm_gripper"]) :] == SLOT_PAD)
    assert jnp.array_equal(emb_mask, jnp.ones((2,), dtype=jnp.bool_))


def test_prepare_head_inputs_returns_none_for_missing_head():
    batch = {"action": {}, "embodiment": {}}
    assert prepare_head_inputs(batch, "single", 2, 2, "xarm_gripper") is None


def test_prepare_head_inputs_rejects_oversized_actions():
    batch = {
        "action": {
            "single": jnp.ones((2, 7, 11), dtype=jnp.float32),
        },
        "embodiment": {},
    }

    with pytest.raises(ValueError, match="exceeds bounds"):
        prepare_head_inputs(batch, "single", max_h=6, max_a=10, embodiment_name="xarm_gripper")
