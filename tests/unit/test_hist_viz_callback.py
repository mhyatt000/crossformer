from __future__ import annotations

import numpy as np

from crossformer.data.grain.metadata import ArrayStatistics, DatasetStatistics
from crossformer.embody import DOF
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer, HistVizCallback
from crossformer.utils.jax_utils import str2jax
import wandb


def _stats(mean, std, mask=None):
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mask is None:
        mask = np.ones_like(mean, dtype=bool)
    return ArrayStatistics(
        mean=mean,
        std=std,
        minimum=np.zeros_like(mean),
        maximum=np.ones_like(mean),
        mask=np.asarray(mask, dtype=bool),
    )


def _names(*xs: str) -> np.ndarray:
    width = max(len(x) for x in xs)
    out = np.zeros((len(xs), width), dtype=np.uint8)
    for i, x in enumerate(xs):
        enc = np.array(str2jax(x))
        out[i, : len(enc)] = enc
    return out


def test_hist_viz_callback_unnormalizes_batch_and_flat_predict():
    cb = HistVizCallback(
        stats={
            "ds_joint": DatasetStatistics(
                action={
                    "joints": _stats([10, 20], [2, 5]),
                    "gripper": _stats([7], [1], mask=[False]),
                },
                proprio={},
                num_transitions=0,
                num_trajectories=0,
            ),
            "ds_pose": DatasetStatistics(
                action={
                    "pose": _stats([100, 200, 300, 400, 500, 600], [1, 1, 1, 10, 20, 30]),
                },
                proprio={},
                num_transitions=0,
                num_trajectories=0,
            ),
        }
    )

    batch = {
        "act": {
            "base": np.array(
                [
                    [[[1.0, -1.0, 0.5]]],
                    [[[2.0, 3.0, 0.25]]],
                ],
                dtype=np.float32,
            ),
            "id": np.array(
                [
                    [DOF["j0"], DOF["j1"], DOF["gripper"]],
                    [DOF["ee_x"], DOF["ee_ry"], DOF["gripper"]],
                ],
                dtype=np.int32,
            ),
        },
        "info": {
            "dataset_name": _names("ds_joint", "ds_pose"),
        },
    }
    predict = {
        "predict": np.array(
            [
                [[0.0, 1.0, 0.1]],
                [[4.0, 5.0, 0.2]],
            ],
            dtype=np.float32,
        )
    }

    batch_vals = cb.denorm.denormalize(batch["act"]["base"], batch["act"]["id"], ["ds_joint", "ds_pose"])
    pred_vals = cb.denorm.denormalize(predict["predict"], batch["act"]["id"], ["ds_joint", "ds_pose"], horizon=1)

    assert np.allclose(batch_vals["j0"], np.array([12.0], dtype=np.float32))
    assert np.allclose(batch_vals["j1"], np.array([15.0], dtype=np.float32))
    assert np.allclose(batch_vals["ee_x"], np.array([102.0], dtype=np.float32))
    assert np.allclose(batch_vals["ee_ry"], np.array([560.0], dtype=np.float32))
    assert np.allclose(batch_vals["gripper"], np.array([0.5, 0.25], dtype=np.float32))

    assert np.allclose(pred_vals["j0"], np.array([10.0], dtype=np.float32))
    assert np.allclose(pred_vals["j1"], np.array([25.0], dtype=np.float32))
    assert np.allclose(pred_vals["ee_x"], np.array([104.0], dtype=np.float32))
    assert np.allclose(pred_vals["ee_ry"], np.array([600.0], dtype=np.float32))
    assert np.allclose(pred_vals["gripper"], np.array([0.1, 0.2], dtype=np.float32))

    out = cb(batch, predict)

    assert isinstance(out["data"]["j0"], wandb.Histogram)
    assert isinstance(out["predict"]["ee_x"], wandb.Histogram)


def test_action_batch_denormalizer_round_trip_denorm_norm_denorm():
    stats = {
        "ds_joint": DatasetStatistics(
            action={
                "joints": _stats([10, 20], [2, 5]),
                "gripper": _stats([7], [1], mask=[False]),
            },
            proprio={},
            num_transitions=0,
            num_trajectories=0,
        ),
    }
    denorm = ActionBatchDenormalizer(stats)

    norm_batch = np.array(
        [
            [
                [[1.0, -1.0, 0.5], [0.0, 2.0, 0.25]],
                [[-2.0, 1.5, 0.75], [3.0, -0.5, 0.0]],
            ],
        ],
        dtype=np.float32,
    )
    dof_ids = np.array([[DOF["j0"], DOF["j1"], DOF["gripper"]]], dtype=np.int32)

    raw = denorm.denormalize(norm_batch, dof_ids, ["ds_joint"])
    action_stats = denorm._action_stats("ds_joint")
    expected_norm = {
        "j0": norm_batch[..., 0].reshape(-1),
        "j1": norm_batch[..., 1].reshape(-1),
        "gripper": norm_batch[..., 2].reshape(-1),
    }

    renorm = {}
    redone = {}
    for dof_id in dof_ids[0]:
        dof_name = denorm._dof_name(int(dof_id))
        vals = raw[dof_name][:, None]
        stat = denorm._dof_array_stats(action_stats, dof_name)
        if stat is None:
            renorm[dof_name] = vals[:, 0]
            redone[dof_name] = vals[:, 0]
            continue
        renorm[dof_name] = stat.normalize(vals)[:, 0]
        redone[dof_name] = stat.unnormalize(renorm[dof_name][:, None])[:, 0]

    for dof_name, vals in raw.items():
        np.testing.assert_allclose(renorm[dof_name], expected_norm[dof_name])
        np.testing.assert_allclose(redone[dof_name], vals)
