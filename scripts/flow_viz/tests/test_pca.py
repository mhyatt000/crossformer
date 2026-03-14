from __future__ import annotations

from pathlib import Path
import numpy as np

from crossformer.data.grain.datasets import _DecodedArrayRecord
from scripts.flow_viz.flow_viz_pca import _plot_two_panel_video, _ensure_snd


def test_flow_viz_pca_smoke():
    shards = sorted(Path("~/crossformer_data").expanduser().glob("**/*.arrayrecord"))
    records = _DecodedArrayRecord(shards)
    assert len(records) > 0, "No arrayrecords found."

    step = records[0]

    q_raw = step.get("action", {}).get("q", None)
    if q_raw is not None:
        q = np.asarray(q_raw, dtype=np.float32)
        if q.ndim == 3:
            q = q[:, 0, :]
    else:
        k3 = np.asarray(step["action"]["k3ds"], dtype=np.float32)  # [S,J,4]
        q = k3[:, :, :3].reshape(k3.shape[0], -1)                  # [S,D]

    x = np.tanh(q).astype(np.float32)

    q_flow = _ensure_snd(q, "q_flow")
    x_flow = _ensure_snd(x, "x_flow")

    out_dir = Path("/tmp/flow_viz_tests/pca")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_two_panel_video(
        q_flow=q_flow,
        x_flow=x_flow,
        out_png=out_dir / "pca_step0.png",
        out_mp4=out_dir / "pca.mp4",
        fps=8,
    )

    assert (out_dir / "pca_step0.png").exists()
    assert (out_dir / "pca.mp4").exists()
