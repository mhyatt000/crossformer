from __future__ import annotations

import numpy as np

from crossformer.data.grain.embody import build_action_norm_mask
from crossformer.data.grain.metadata import OnlineStats
from crossformer.embody import GRIPPER, SINGLE


def test_build_action_norm_mask_uses_bodypart_masks():
    action = {
        "joints": np.zeros((4, 7), dtype=np.float32),
        "gripper": np.zeros((4, 1), dtype=np.float32),
        "position": np.zeros((4, 3), dtype=np.float32),
        "orientation": np.zeros((4, 3), dtype=np.float32),
    }

    mask = build_action_norm_mask(action, SINGLE)

    assert np.all(mask["joints"])
    assert np.array_equal(mask["gripper"], np.array([False]))
    assert np.all(mask["position"])
    assert np.all(mask["orientation"])
    assert GRIPPER.action_norm_mask == (False,)


def test_online_stats_ignores_masked_dims():
    stats = OnlineStats((2,), dtype=np.float64)
    stats.update(np.array([1.0, 10.0]), np.array([True, False]))
    stats.update(np.array([3.0, 20.0]), np.array([True, False]))

    out = stats.finalize()

    np.testing.assert_allclose(out["mean"], np.array([2.0, 0.0]))
    np.testing.assert_allclose(out["std"], np.array([1.0, 0.0]))
    np.testing.assert_allclose(out["minimum"], np.array([1.0, 0.0]))
    np.testing.assert_allclose(out["maximum"], np.array([3.0, 0.0]))
    np.testing.assert_array_equal(out["mask"], np.array([True, False]))
