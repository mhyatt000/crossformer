from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.data.grain.metadata import ArrayStatistics, DatasetStatistics
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer
from scripts.train.xflow import (
    adapt_rast_batch,
    Config,
    denorm_canonical,
    denorm_joints,
    make_model_config,
    normalize_obs,
    RAST_IDS,
    resolve_obs_keys,
)


def _stats(mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return ArrayStatistics(
        mean=mean,
        std=std,
        minimum=np.zeros_like(mean),
        maximum=np.ones_like(mean),
        mask=np.ones_like(mean, dtype=bool),
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


def test_denorm_joints_unnormalizes_joint_array():
    denorm = ActionBatchDenormalizer(
        {
            "ds_joint": DatasetStatistics(
                action={"joints": _stats([1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8])},
                proprio={},
                num_transitions=0,
                num_trajectories=0,
            )
        }
    )
    arr = np.array([[0, 1, -1, 0.5, -0.5, 2, -2]], dtype=np.float32)

    out = denorm_joints(arr, denorm, "ds_joint")

    expected = np.array([[1, 5, -1, 6.5, 2, 20, -9]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_denorm_joints_requires_stats():
    arr = np.zeros((1, 7), dtype=np.float32)

    with pytest.raises(ValueError, match=r"ActionBatchDenormalizer\.stats is required"):
        denorm_joints(arr, ActionBatchDenormalizer(), "ds_joint")


def test_adapt_rast_batch_keeps_gripper_slot():
    act = {
        "base": np.array([[[10.0, 20.0, 0.25]]], dtype=np.float32),
        "id": np.array([[RAST_IDS[0], RAST_IDS[1], RAST_IDS[-1]]], dtype=np.int32),
    }
    flow = np.array([[[[[11.0, 21.0, 0.75]]]]], dtype=np.float32)

    out, keep = adapt_rast_batch(act, flow)

    assert keep.tolist() == [0]
    np.testing.assert_allclose(out["act"]["base"][0, 0, 0, [0, 1, 7]], np.array([10.0, 20.0, 0.25], np.float32))
    np.testing.assert_allclose(out["predict"][0, 0, 0, 0, [0, 1, 7]], np.array([11.0, 21.0, 0.75], np.float32))


def test_denorm_canonical_respects_explicit_dof_ids():
    denorm = ActionBatchDenormalizer(
        {
            "ds_joint": DatasetStatistics(
                action={
                    "joints": _stats([1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8]),
                    "gripper": _stats([0.4], [0.1]),
                },
                proprio={},
                num_transitions=0,
                num_trajectories=0,
            )
        }
    )
    arr = np.array([[0, 1, -1, 0.5, -0.5, 2, -2, 0.25]], dtype=np.float32)

    out = denorm_canonical(arr, denorm, "ds_joint", np.asarray(RAST_IDS))

    expected = np.array([[1, 5, -1, 6.5, 2, 20, -9, 0.25]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)
