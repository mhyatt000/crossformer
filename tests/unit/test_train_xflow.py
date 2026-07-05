from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from crossformer.data.grain.metadata import ArrayStatistics, DatasetStatistics
from crossformer.model.components.heads.dof import CHUNK_PAD
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.tokenizers import LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.utils.callbacks.adapt import adapt_rast_batch, denorm_canonical, JOINT_IDS, RAST_IDS
from crossformer.utils.callbacks.base import extract_bundled_actions, flatten_obs
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer


def _spec(cls: type[object], **kwargs: object) -> dict[str, Any]:
    return {"module": cls.__module__, "name": cls.__name__, "args": (), "kwargs": kwargs}


def _stats(mean: list[float] | np.ndarray, std: list[float] | np.ndarray) -> ArrayStatistics:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    kwargs = {
        "mean": mean,
        "std": std,
        "minimum": np.zeros_like(mean),
        "maximum": np.ones_like(mean),
        "mask": np.ones_like(mean, dtype=bool),
    }
    return ArrayStatistics(**kwargs)


def test_minimal_model_config_wires_xflow_bounds() -> None:
    token_dim, transformer_kwargs = common_transformer_sizes("dummy")
    model_cfg = {
        "observation_tokenizers": {
            "proprio": _spec(LowdimObsTokenizer, obs_keys=("proprio",)),
        },
        "task_tokenizers": {},
        "heads": {
            "action": _spec(
                XFlowHead,
                readout_key="readout_action",
                max_horizon=6,
                max_dofs=9,
            )
        },
        "readouts": {"action": 4},
        "transformer_kwargs": transformer_kwargs,
        "token_embedding_size": token_dim,
        "max_horizon": 11,
    }
    head_cfg = model_cfg["heads"]["action"]
    head_kwargs = head_cfg["kwargs"]

    assert model_cfg["max_horizon"] == 11
    assert model_cfg["readouts"] == {"action": 4}
    assert head_cfg["name"] == "XFlowHead"
    assert head_kwargs["max_horizon"] == 6
    assert head_kwargs["max_dofs"] == 9
    assert head_kwargs["readout_key"] == "readout_action"


def test_flatten_obs_adds_channel_and_flattens() -> None:
    obs = {
        "scalar": np.ones((2, 3)),
        "pose": np.ones((2, 3, 2, 4)),
        "already_seq": np.ones((2, 3, 5)),
    }

    out = flatten_obs(obs, ("scalar", "pose", "already_seq"))

    assert out["scalar"].shape == (2, 3, 1)
    assert out["pose"].shape == (2, 3, 8)
    assert out["already_seq"].shape == (2, 3, 5)


def test_extract_bundled_actions_uses_horizon_mask() -> None:
    batch = {
        "act": {
            "base": np.zeros((2, 3, 4), dtype=np.float32),
            "id": np.ones((2, 4), dtype=np.int32),
        },
        "mask": {
            "horizon": np.array(
                [
                    [True, True, False],
                    [True, False, False],
                ]
            )
        },
    }

    _actions, _dof_ids, chunk_steps, _view_ids, _mask_act = extract_bundled_actions(batch, max_h=3)

    np.testing.assert_allclose(
        np.asarray(chunk_steps),
        np.array(
            [
                [0.0, 1.0, CHUNK_PAD],
                [0.0, CHUNK_PAD, CHUNK_PAD],
            ],
            dtype=np.float32,
        ),
    )


def test_denorm_canonical_unnormalizes_joint_array() -> None:
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

    out = denorm_canonical(arr, denorm, "ds_joint", np.asarray(JOINT_IDS))

    expected = np.array([[1, 5, -1, 6.5, 2, 20, -9]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_denorm_canonical_requires_stats() -> None:
    arr = np.zeros((1, 7), dtype=np.float32)

    with pytest.raises(ValueError, match=r"ActionBatchDenormalizer\.stats is required"):
        denorm_canonical(arr, ActionBatchDenormalizer(), "ds_joint", np.asarray(JOINT_IDS))


def test_adapt_rast_batch_keeps_gripper_slot() -> None:
    act = {
        "base": np.array([[[10.0, 20.0, 0.25]]], dtype=np.float32),
        "id": np.array([[RAST_IDS[0], RAST_IDS[1], RAST_IDS[-1]]], dtype=np.int32),
    }
    flow = np.array([[[[[11.0, 21.0, 0.75]]]]], dtype=np.float32)

    out, keep = adapt_rast_batch(act, flow)

    assert keep.tolist() == [0]
    np.testing.assert_allclose(out["act"]["base"][0, 0, 0, [0, 1, 7]], np.array([10.0, 20.0, 0.25], np.float32))
    np.testing.assert_allclose(out["predict"][0, 0, 0, 0, [0, 1, 7]], np.array([11.0, 21.0, 0.75], np.float32))


def test_denorm_canonical_respects_explicit_dof_ids() -> None:
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
