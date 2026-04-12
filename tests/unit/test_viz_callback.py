from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from crossformer.viz.flow_pca import compute_fk, fit_pca, prep_data, render_frames

needs_cusolver = pytest.mark.integration

# Dimensions
S, A, F = 50, 7, 5


def _fake_fk(q: np.ndarray) -> np.ndarray:
    """Trivial FK: first 3 joints are xyz."""
    return q[:, :3].astype(np.float32)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def base_and_flow(rng):
    base = rng.standard_normal((S, A)).astype(np.float32)
    # flow converges from noise toward base
    noise = rng.standard_normal((F, S, A)).astype(np.float32)
    alphas = np.linspace(0, 1, F)[:, None, None]
    flow = (1 - alphas) * noise + alphas * base[None]
    return base, flow


# ── prep_data ──────────────────────────────────────────────────


class TestPrepData:
    def test_shapes(self, base_and_flow):
        base, flow = base_and_flow
        base_sub, flow_sub = prep_data(base, flow, max_pts=30)
        assert base_sub.shape == (30, A)
        assert flow_sub.shape == (F, 30, A)

    def test_subsample_capped(self, base_and_flow):
        base, flow = base_and_flow
        base_sub, _ = prep_data(base, flow, max_pts=S + 100)
        assert base_sub.shape[0] == S

    def test_no_nans(self, base_and_flow):
        base, flow = base_and_flow
        base_sub, flow_sub = prep_data(base, flow)
        assert not np.any(np.isnan(base_sub))
        assert not np.any(np.isnan(flow_sub))


# ── compute_fk ─────────────────────────────────────────────────


class TestComputeFK:
    def test_shapes(self, base_and_flow):
        base, flow = base_and_flow
        base_fk, flow_fk = compute_fk(base, flow, _fake_fk)
        assert base_fk.shape == (S, 3)
        assert flow_fk.shape == (F, S, 3)

    def test_matches_manual(self, rng):
        base = rng.standard_normal((10, A)).astype(np.float32)
        flow = rng.standard_normal((3, 10, A)).astype(np.float32)
        base_fk, flow_fk = compute_fk(base, flow, _fake_fk)
        np.testing.assert_allclose(base_fk, base[:, :3])
        np.testing.assert_allclose(flow_fk[0], flow[0, :, :3])


# ── fit_pca ────────────────────────────────────────────────────


@needs_cusolver
class TestFitPCA:
    def test_output_types_and_shapes(self, base_and_flow):
        base, _ = base_and_flow
        base_fk = _fake_fk(base)
        _joint_st, _fk_st, base_j2d, base_fk2d, _jlim, _flim = fit_pca(base, base_fk)
        assert base_j2d.shape == (S, 2)
        assert base_fk2d.shape == (S, 2)

    def test_limits_pad(self, base_and_flow):
        base, _ = base_and_flow
        base_fk = _fake_fk(base)
        _, _, base_j2d, _, jlim, _ = fit_pca(base, base_fk)
        assert jlim[0][0] == pytest.approx(float(base_j2d[:, 0].min()) - 0.5)
        assert jlim[0][1] == pytest.approx(float(base_j2d[:, 0].max()) + 0.5)

    def test_without_joint_data(self, base_and_flow):
        base, _ = base_and_flow
        base_fk = _fake_fk(base)
        joint_st, _fk_st, base_j2d, base_fk2d, jlim, _flim = fit_pca(None, base_fk)
        assert joint_st is None
        assert base_j2d.shape == (0, 2)
        assert base_fk2d.shape == (S, 2)
        assert jlim == ((-1.0, 1.0), (-1.0, 1.0))


# ── render_frames ──────────────────────────────────────────────


@needs_cusolver
class TestRenderFrames:
    def test_shape_and_dtype(self, base_and_flow):
        base, flow = base_and_flow
        base_fk = _fake_fk(base)
        flow_fk = compute_fk(base, flow, _fake_fk)[1]
        joint_st, fk_st, base_j2d, base_fk2d, jlim, flim = fit_pca(base, base_fk)

        frames = render_frames(
            flow,
            flow_fk,
            joint_st,
            fk_st,
            base_j2d,
            base_fk2d,
            jlim,
            flim,
            figsize=(4.0, 2.0),
        )
        assert frames.ndim == 4
        assert frames.shape[0] == F
        assert frames.shape[3] == 3
        assert frames.dtype == np.uint8

    def test_frames_not_identical(self, base_and_flow):
        base, flow = base_and_flow
        base_fk = _fake_fk(base)
        flow_fk = compute_fk(base, flow, _fake_fk)[1]
        joint_st, fk_st, base_j2d, base_fk2d, jlim, flim = fit_pca(base, base_fk)

        frames = render_frames(
            flow,
            flow_fk,
            joint_st,
            fk_st,
            base_j2d,
            base_fk2d,
            jlim,
            flim,
            figsize=(4.0, 2.0),
        )
        assert not np.array_equal(frames[0], frames[-1])

    def test_supports_human_xyz_overlay_without_robot_joint_panel(self, base_and_flow):
        base, flow = base_and_flow
        base_fk = _fake_fk(base)
        human_flow = flow[:, :, :3]
        _joint_st, fk_st, base_j2d, base_fk2d, jlim, flim = fit_pca(None, base_fk)

        frames = render_frames(
            None,
            None,
            None,
            fk_st,
            base_j2d,
            base_fk2d,
            jlim,
            flim,
            figsize=(4.0, 2.0),
            human_flow_xyz=human_flow,
        )
        assert frames.ndim == 4
        assert frames.shape[0] == F
        assert frames.shape[3] == 3


# ── VizCallback.save ───────────────────────────────────────────


class TestVizCallbackSave:
    def test_save_gif(self, tmp_path):
        from crossformer.utils.callbacks.viz import VizCallback

        cb = VizCallback()
        frames = np.random.default_rng(0).integers(0, 255, (3, 48, 64, 3), dtype=np.uint8)
        out = cb.save(frames, tmp_path / "test.gif")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_bad_ext(self, tmp_path):
        from crossformer.utils.callbacks.viz import VizCallback

        cb = VizCallback()
        frames = np.zeros((2, 10, 10, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected"):
            cb.save(frames, tmp_path / "test.png")


# ── VizCallback selection helpers ──────────────────────────────


class TestVizCallbackSelect:
    def test_select_base_truncates(self):
        from crossformer.utils.callbacks.viz import VizCallback

        cb = VizCallback(joint_dim=7)
        arr = np.zeros((4, 10))
        result = cb._select_base(arr)
        assert result.shape == (4, 7)

    def test_select_base_rejects_small_dim(self):
        from crossformer.utils.callbacks.viz import VizCallback

        cb = VizCallback(joint_dim=7)
        arr = np.zeros((4, 3))
        with pytest.raises(ValueError, match="joint dim"):
            cb._select_base(arr)

    def test_select_flow_rejects_1d(self):
        from crossformer.utils.callbacks.viz import VizCallback

        cb = VizCallback()
        with pytest.raises(ValueError, match="ndim"):
            cb._select_flow(np.zeros(5))

    def test_get_nested(self):
        from crossformer.utils.callbacks.viz import VizCallback

        cb = VizCallback()
        batch = {"a": {"b": {"c": 42}}}
        assert cb._get(batch, ("a", "b", "c")) == 42
