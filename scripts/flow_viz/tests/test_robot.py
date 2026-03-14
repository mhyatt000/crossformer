from __future__ import annotations

from pathlib import Path
import numpy as np

from xgym import calibrate
from xgym.calibrate.urdf.robot import urdf

from scripts.flow_viz.flow_viz_robot import (
    make_robot_from_urdf,
    render_robot_q_flow_video,
)


def test_flow_viz_robot_smoke():
    S, Dq = 30, 7
    t = np.linspace(0, 2 * np.pi, S, dtype=np.float32)
    q_steps = np.stack(
        [
            0.4 * np.sin(t),
            0.3 * np.cos(0.7 * t),
            0.2 * np.sin(1.3 * t),
            0.3 * np.cos(1.1 * t),
            0.2 * np.sin(0.9 * t),
            0.1 * np.cos(1.5 * t),
            0.2 * np.sin(0.5 * t),
        ],
        axis=1,
    ).astype(np.float32)

    robot = make_robot_from_urdf(
        urdf_path=urdf,
        mesh_dir=calibrate.urdf.robot.DNAME / "assets",
    )

    out_dir = Path("/tmp/flow_viz_tests/robot")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = render_robot_q_flow_video(
        robot=robot,
        q_steps=q_steps,
        out_mp4=out_dir / "robot.mp4",
        out_png=out_dir / "robot_step0.png",
        fps=10,
    )

    assert Path(result["out_mp4"]).exists()
    assert Path(result["out_png"]).exists()
