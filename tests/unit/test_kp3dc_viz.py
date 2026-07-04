from __future__ import annotations

import numpy as np
import wandb

from crossformer.utils.callbacks.base import EvalContext
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer
from crossformer.utils.callbacks.kp3dc_viz import Kp3dcVizCallback, KP3DC_IDS, project_kp3dc
from crossformer.utils.jax_utils import str2np

H, N_VIEWS = 4, 2


def test_project_kp3dc_matches_manual_pinhole() -> None:
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    xyz = np.array([[0.1, -0.2, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    uv = project_kp3dc(xyz, K)
    np.testing.assert_allclose(uv[0], [100 * 0.1 / 2 + 32, 100 * -0.2 / 2 + 24], rtol=1e-5)
    np.testing.assert_allclose(uv[1], [32.0, 24.0], rtol=1e-5)


def test_project_kp3dc_nan_behind_camera() -> None:
    K = np.eye(3, dtype=np.float32)
    uv = project_kp3dc(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), K)
    assert np.all(np.isnan(uv))


def _names(*xs: str) -> np.ndarray:
    width = max(len(x) for x in xs)
    out = np.zeros((len(xs), width), dtype=np.uint8)
    for i, x in enumerate(xs):
        out[i, : len(str2np(x))] = str2np(x)
    return out


def _kp_batch() -> dict:
    """One sample whose slots are the full 42-dim kp3dc block, duplicated per view."""
    n = len(KP3DC_IDS)
    dof_ids = np.tile(np.asarray(KP3DC_IDS, dtype=np.int32), N_VIEWS)[None]  # (1, V*42)
    view_ids = np.repeat(np.arange(1, N_VIEWS + 1, dtype=np.int32), n)[None]
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-0.3, 0.3, size=(H, N_VIEWS * n)).astype(np.float32)
    xyz[:, 2::3] = rng.uniform(0.5, 1.5, size=(H, N_VIEWS * n // 3))  # z > 0
    K = np.array([[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return {
        "act": {"base": xyz[None, None], "id": dof_ids, "view": view_ids},  # base (1, 1, H, A)
        "observation": {"image": np.zeros((1, 1, N_VIEWS, 16, 16, 3), dtype=np.uint8)},
        "state": {"intr": {"K": np.broadcast_to(K, (1, 1, N_VIEWS, 3, 3)).copy()}},
        "info": {"dataset_name": _names("dummy_ds")},
    }


def _ctx(batch: dict) -> EvalContext:
    ctx = EvalContext(
        model=None,
        params=None,
        rng=None,
        step=0,
        batch=batch,
        denorm=ActionBatchDenormalizer({"dummy_ds": {"action": {}}}),
    )
    ctx.__dict__["pred"] = np.asarray(batch["act"]["base"])
    return ctx


def test_kp3dc_callback_renders_one_image_per_view() -> None:
    out = Kp3dcVizCallback(every=1)(_ctx(_kp_batch()))
    assert sorted(out) == [f"view_{v}" for v in range(N_VIEWS)]
    assert all(isinstance(v, wandb.Image) for v in out.values())


def test_kp3dc_callback_skips_without_state() -> None:
    batch = _kp_batch()
    del batch["state"]
    assert Kp3dcVizCallback(every=1)(_ctx(batch)) == {}
