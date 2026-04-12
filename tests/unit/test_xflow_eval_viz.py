from __future__ import annotations

import numpy as np

from crossformer.run.xflow_eval import adapt_viz_batch, JOINT_IDS, POS_IDS


def test_adapt_viz_batch_splits_robot_and_human_tracks():
    batch = {
        "act": {
            "base": np.array(
                [
                    [[[1.0, 2.0, 3.0, 4.0, 5.0]]],
                    [[[10.0, 20.0, 30.0, 0.0, 0.0]]],
                ],
                dtype=np.float32,
            ),
            "id": np.array(
                [
                    [JOINT_IDS[0], JOINT_IDS[1], POS_IDS[0], POS_IDS[1], POS_IDS[2]],
                    [POS_IDS[0], POS_IDS[1], POS_IDS[2], 0, 0],
                ],
                dtype=np.int32,
            ),
        },
        "mask": {
            "embodiment": {
                "single": np.array([[True], [False]]),
                "human_single": np.array([[False], [True]]),
            }
        },
    }
    flow = np.array(
        [
            [
                [[[3.0, 4.0, 6.0, 7.0, 8.0]]],
                [[[11.0, 21.0, 31.0, 0.0, 0.0]]],
            ],
            [
                [[[5.0, 6.0, 9.0, 10.0, 11.0]]],
                [[[12.0, 22.0, 32.0, 0.0, 0.0]]],
            ],
        ],
        dtype=np.float32,
    )

    out, keep = adapt_viz_batch(batch, flow)

    assert keep["robot"].tolist() == [0]
    assert keep["robot_xyz"].tolist() == [0]
    assert keep["human"].tolist() == [1]
    np.testing.assert_allclose(out["act"]["base"][0, 0, 0, :2], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(out["predict"][1, 0, 0, 0, :2], np.array([5.0, 6.0], dtype=np.float32))
    np.testing.assert_allclose(out["robot_xyz"]["base"][0, 0, 0], np.array([3.0, 4.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(out["robot_xyz"]["predict"][0, 0, 0, 0], np.array([6.0, 7.0, 8.0], dtype=np.float32))
    np.testing.assert_allclose(out["human_xyz"]["base"][0, 0, 0], np.array([10.0, 20.0, 30.0], dtype=np.float32))
    np.testing.assert_allclose(out["human_xyz"]["predict"][0, 0, 0, 0], np.array([11.0, 21.0, 31.0], dtype=np.float32))
