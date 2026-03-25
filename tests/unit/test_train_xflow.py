from __future__ import annotations

import jax.numpy as jnp
import pytest

from scripts.train.xflow import (
    Config,
    make_model_config,
    normalize_obs,
    resolve_obs_keys,
)


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
