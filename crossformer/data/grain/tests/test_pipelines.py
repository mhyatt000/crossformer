"""Unit tests for the Grain data pipeline."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from crossformer.data.grain import builders, make_interleaved_dataset, make_single_dataset


def _make_trajectory(length: int, *, language: str, offset: float = 0.0) -> dict:
    return {
        "observation": {
            "rgb": np.array([f"img_{i}" for i in range(length)], dtype=object),
            "proprio": np.stack([
                np.linspace(0.0 + offset, 1.0 + offset, num=2, dtype=np.float32)
                for _ in range(length)
            ]),
        },
        "action": np.stack(
            [
                np.array([i + offset, i + 1 + offset], dtype=np.float32)
                for i in range(length)
            ]
        ),
        "language": np.array([language] * length, dtype=object),
    }


def _make_config(name: str, *, trajectories: list[dict], tmp_path: Path) -> builders.GrainDatasetConfig:
    return builders.GrainDatasetConfig(
        name=name,
        source=trajectories,
        image_obs_keys={"main": "rgb"},
        depth_obs_keys={},
        proprio_obs_keys={"arm": "proprio"},
        proprio_obs_dims={"arm": 2},
        language_key="language",
        statistics_save_dir=str(tmp_path / f"stats_{name}"),
    )


class GrainPipelineTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_single_dataset_chunking(self):
        trajectories = [
            _make_trajectory(3, language="pick"),
            _make_trajectory(2, language="", offset=0.5),
        ]
        config = _make_config("toy", trajectories=trajectories, tmp_path=self.tmp_path)
        result = make_single_dataset(
            config,
            train=False,
            traj_transform_kwargs={
                "window_size": 2,
                "action_horizon": 2,
                "skip_unlabeled": True,
            },
        )

        frames = list(result.dataset)
        # Only the labeled trajectory remains after skip_unlabeled.
        self.assertEqual(len(frames), 3)
        for frame in frames:
            self.assertEqual(frame["action"].shape, (2, 2, 2))
            self.assertEqual(frame["action_pad_mask"].shape, (2, 2, 2))
            self.assertEqual(frame["observation"]["image_main"].shape, (2,))
            self.assertEqual(frame["observation"]["timestep_pad_mask"].shape, (2,))
            self.assertEqual(frame["observation"]["task_completed"].shape, (2, 2))
            self.assertEqual(frame["dataset_name"], "toy")

        stats = result.statistics
        self.assertEqual(stats.num_trajectories, 2)
        self.assertEqual(stats.action.mean.shape, (2,))

    def test_interleaved_dataset_sampling(self):
        config_a = _make_config(
            "dataset_a",
            trajectories=[_make_trajectory(2, language="alpha")],
            tmp_path=self.tmp_path,
        )
        config_b = _make_config(
            "dataset_b",
            trajectories=[_make_trajectory(4, language="beta", offset=0.7)],
            tmp_path=self.tmp_path,
        )

        result = make_interleaved_dataset(
            [config_a, config_b],
            train=False,
            sample_weights=[0.8, 0.2],
            shuffle_buffer_size=3,
        )

        frames = list(itertools.islice(result.dataset, 6))
        names = {frame["dataset_name"] for frame in frames}
        self.assertTrue(names.issubset({"dataset_a", "dataset_b"}))
        self.assertEqual(set(result.statistics.keys()), {"dataset_a", "dataset_b"})


if __name__ == "__main__":
    unittest.main()
