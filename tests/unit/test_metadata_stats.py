"""Tests that ArrayStatistics computes per-timestep (shape A) stats and broadcasts to chunks."""

from __future__ import annotations

import numpy as np
import pytest

from crossformer.data.grain.metadata import ArrayStatistics, OnlineStats

# ---------------------------------------------------------------------------
# ArrayStatistics.compute — should reduce chunk dim
# ---------------------------------------------------------------------------


class TestArrayStatisticsCompute:
    def test_3d_input_yields_1d_stats(self):
        """Given (B, H, A) input, stats should have shape (A,)."""
        rng = np.random.default_rng(0)
        B, H, A = 50, 10, 7
        arr = rng.standard_normal((B, H, A))

        stats = ArrayStatistics.compute(arr)

        assert stats.mean.shape == (A,)
        assert stats.std.shape == (A,)
        assert stats.minimum.shape == (A,)
        assert stats.maximum.shape == (A,)
        assert stats.mask.shape == (A,)
        assert stats.p99.shape == (A,)
        assert stats.p01.shape == (A,)

    def test_uses_first_timestep_only(self):
        """Stats should match manual computation on arr[:, 0]."""
        rng = np.random.default_rng(42)
        B, H, A = 100, 5, 3
        arr = rng.standard_normal((B, H, A))

        stats = ArrayStatistics.compute(arr)
        first = arr[:, 0]

        np.testing.assert_allclose(stats.mean, first.mean(axis=0))
        np.testing.assert_allclose(stats.std, first.std(axis=0))
        np.testing.assert_allclose(stats.minimum, first.min(axis=0))
        np.testing.assert_allclose(stats.maximum, first.max(axis=0))

    def test_2d_input_unchanged(self):
        """2D input (B, A) should pass through without slicing."""
        rng = np.random.default_rng(1)
        B, A = 30, 4
        arr = rng.standard_normal((B, A))

        stats = ArrayStatistics.compute(arr)
        assert stats.mean.shape == (A,)
        np.testing.assert_allclose(stats.mean, arr.mean(axis=0))


# ---------------------------------------------------------------------------
# normalize / unnormalize — broadcast (A,) stats to (H, A) input
# ---------------------------------------------------------------------------


class TestBroadcastNormalize:
    @pytest.fixture()
    def stats_and_data(self):
        rng = np.random.default_rng(7)
        A = 5
        stats = ArrayStatistics(
            mean=rng.standard_normal(A),
            std=np.abs(rng.standard_normal(A)) + 0.1,
            minimum=np.zeros(A),
            maximum=np.ones(A),
            mask=np.ones(A, dtype=bool),
        )
        H = 8
        x = rng.standard_normal((H, A))
        return stats, x

    def test_normalize_broadcasts(self, stats_and_data):
        stats, x = stats_and_data
        y = stats.normalize(x)
        assert y.shape == x.shape
        # Each row should be independently normalized by the same (A,) stats
        for t in range(x.shape[0]):
            expected = (x[t] - stats.mean) / np.maximum(stats.std, 1e-8)
            np.testing.assert_allclose(y[t], expected)

    def test_unnormalize_inverts(self, stats_and_data):
        stats, x = stats_and_data
        np.testing.assert_allclose(stats.unnormalize(stats.normalize(x)), x, atol=1e-12)

    def test_mask_preserves_original(self):
        """Masked-out dims should keep original values after normalize."""
        A = 4
        mask = np.array([True, True, False, False])
        stats = ArrayStatistics(
            mean=np.array([1.0, 2.0, 3.0, 4.0]),
            std=np.array([0.5, 0.5, 0.5, 0.5]),
            minimum=np.zeros(A),
            maximum=np.ones(A),
            mask=mask,
        )
        x = np.ones((3, A)) * 10.0
        y = stats.normalize(x)
        # Masked-out columns unchanged
        np.testing.assert_array_equal(y[:, 2:], x[:, 2:])
        # Masked-in columns normalized
        assert not np.allclose(y[:, :2], x[:, :2])


# ---------------------------------------------------------------------------
# OnlineStats — first-timestep shape
# ---------------------------------------------------------------------------


class TestOnlineStatsShape:
    def test_stats_shape_is_action_dim(self):
        """OnlineStats initialized with (A,) should produce (A,) results."""
        A = 6
        os = OnlineStats((A,))
        rng = np.random.default_rng(3)
        for _ in range(20):
            os.update(rng.standard_normal(A))
        result = os.finalize()
        assert result["mean"].shape == (A,)
        assert result["std"].shape == (A,)

    def test_online_matches_batch(self):
        """OnlineStats on first timesteps should match ArrayStatistics.compute."""
        rng = np.random.default_rng(99)
        B, H, A = 200, 4, 3
        arr = rng.standard_normal((B, H, A))

        # batch
        batch_stats = ArrayStatistics.compute(arr)

        # online (simulating what compute_dataset_statistics does)
        os = OnlineStats((A,))
        for i in range(B):
            os.update(arr[i, 0])
        result = os.finalize()

        np.testing.assert_allclose(result["mean"], batch_stats.mean, atol=1e-10)
        # population std
        np.testing.assert_allclose(result["std"], batch_stats.std, atol=1e-10)
