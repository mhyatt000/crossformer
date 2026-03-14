from __future__ import annotations

from pathlib import Path
import numpy as np

from crossformer.data.grain.datasets import _DecodedArrayRecord
from crossformer.utils.callbacks.flow_viz import load_camera_extrinsics, _DEFAULT_K

from scripts.flow_viz.flow_viz_overlay import render_xyz_overlay_video


def test_flow_viz_overlay_smoke():
    shards = sorted(Path("~/crossformer_data").expanduser().glob("**/*.arrayrecord"))
    records = _DecodedArrayRecord(shards)
    assert len(records) > 0, "No arrayrecords found."

    step = records[0]
    img = np.asarray(step["observation"]["image"]["low"])
    xyz0 = np.asarray(step["observation"]["proprio"]["k3ds"], dtype=np.float32)[:, :3]  # [J,3]

    S = 20
    noise = np.random.randn(S, *xyz0.shape).astype(np.float32) * 0.003
    xyz_steps = xyz0[None] + noise  # [S,J,3]

    R, t = load_camera_extrinsics("low")
    K = _DEFAULT_K

    out_dir = Path("/tmp/flow_viz_tests/overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = render_xyz_overlay_video(
        image=img,
        xyz_steps=xyz_steps,
        out_mp4=out_dir / "overlay.mp4",
        out_png=out_dir / "overlay_step0.png",
        K=K,
        R=R,
        t=t,
        fps=10,
    )

    assert Path(result["out_mp4"]).exists()
    assert Path(result["out_png"]).exists()
