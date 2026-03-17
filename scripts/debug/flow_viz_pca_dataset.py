from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

from scripts.flow_viz.flow_viz_pca import _ensure_snd, _load_q_from_dataset, _plot_two_panel_video


@dataclass
class Config:
    dataset_dir: Path = Path("~/.cache/arrayrecords/sweep_mano").expanduser()
    traj_index: int = 0
    max_steps: int = 80
    sample_mode: Literal["single", "episodes", "window"] = "episodes"
    num_samples: int = 16
    fps: int = 8
    out_dir: Path = Path("/tmp/flow_viz_pca_debug")


def main(cfg: Config) -> None:
    q = _load_q_from_dataset(
        path=cfg.dataset_dir,
        traj_index=cfg.traj_index,
        max_steps=cfg.max_steps,
        sample_mode=cfg.sample_mode,
        num_samples=cfg.num_samples,
    )
    q_flow = _ensure_snd(q, "q_flow")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = _plot_two_panel_video(
        q_flow=q_flow,
        x_flow=None,
        out_png=cfg.out_dir / "pca_step0.png",
        out_mp4=cfg.out_dir / "pca.mp4",
        fps=cfg.fps,
    )
    print(out)


if __name__ == "__main__":
    main(tyro.cli(Config))
