from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import wandb

from crossformer.viz.flow.overlay import render_xyz_overlay_video
from crossformer.viz.flow.pca import _ensure_snd, _plot_two_panel_video
from crossformer.viz.flow.robot import render_robot_q_flow_video


def render_xyz_overlay(
    xyz_sftj3: np.ndarray,
    img_for_overlay: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    fps: int,
    flow_overlay_ft_index: int,
) -> wandb.Video:
    ft_idx = min(flow_overlay_ft_index, xyz_sftj3.shape[1] - 1)
    xyz_sj3 = xyz_sftj3[:, ft_idx]
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_mp4,
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png,
    ):
        out_mp4 = Path(f_mp4.name)
        out_png = Path(f_png.name)
    render_xyz_overlay_video(
        image=img_for_overlay,
        xyz_steps=xyz_sj3,
        out_mp4=out_mp4,
        out_png=out_png,
        K=K,
        R=R,
        t=t,
        fps=fps,
    )
    return wandb.Video(str(out_mp4), fps=fps, format="mp4")


def render_joint_fk_pca(
    q_sfd: np.ndarray,
    xyz_sftj3: np.ndarray | None,
    xyz_key: str | None,
    pfx: str,
    *,
    fps: int,
    q_pca_dims: int,
    flow_overlay_ft_index: int,
) -> wandb.Image:
    q_flow = q_sfd[..., :q_pca_dims]
    if xyz_sftj3 is None:
        x_flow = q_flow
    else:
        ft_idx = min(flow_overlay_ft_index, q_sfd.shape[1] - 1)
        q_sd = q_sfd[:, ft_idx, :q_pca_dims]
        ft_idx = min(ft_idx, xyz_sftj3.shape[1] - 1)
        x_sd = xyz_sftj3[:, ft_idx].reshape(xyz_sftj3.shape[0], -1)
        q_flow = _ensure_snd(q_sd, "q_flow")
        x_flow = _ensure_snd(x_sd, "x_flow")
    out_png = Path("/tmp/pca_step0.png")
    out_mp4 = Path("/tmp/pca.mp4")
    _plot_two_panel_video(
        q_flow=q_flow,
        x_flow=x_flow,
        out_png=out_png,
        out_mp4=out_mp4,
        fps=fps,
    )
    return wandb.Image(str(out_png), caption=f"{pfx} joint_fk_pca (xyz={xyz_key or 'n/a'})")


def render_robot_flow(
    q_sfd: np.ndarray,
    robot: Any,
    *,
    fps: int,
    flow_overlay_ft_index: int,
    robot_pad_gripper: bool,
) -> wandb.Video:
    ft_idx = min(flow_overlay_ft_index, q_sfd.shape[1] - 1)
    q_sd = q_sfd[:, ft_idx, :]
    with (
        tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_mp4,
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_png,
    ):
        out_mp4 = Path(f_mp4.name)
        out_png = Path(f_png.name)
    render_robot_q_flow_video(
        robot=robot,
        q_steps=q_sd,
        out_mp4=out_mp4,
        out_png=out_png,
        fps=fps,
        pad_gripper=robot_pad_gripper,
    )
    return wandb.Video(str(out_mp4), fps=fps, format="mp4")
