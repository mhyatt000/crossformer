from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import imageio
import numpy as np
import wandb

from crossformer.utils.callbacks.flow_viz import _DEFAULT_K, FlowVisCallback


def _dummy_iter():
    while True:
        yield {"observation": {}}


def _make_eval_step(seed: int, batch_size: int, flow_steps: int, future_steps: int, joints: int):
    rng = np.random.default_rng(seed)
    scales = np.linspace(1.0, 0.05, flow_steps, dtype=np.float32)[None, :, None, None, None]
    w_q2x = rng.normal(size=(7, 3)).astype(np.float32)

    def _eval_step(_state, _batch):
        target = rng.standard_normal((batch_size, future_steps, joints, 3)).astype(np.float32)
        noise = rng.standard_normal((batch_size, flow_steps, future_steps, joints, 3)).astype(np.float32)
        pred_steps = target[:, None] + 0.2 * scales * noise
        q_target = rng.standard_normal((batch_size, future_steps, 7)).astype(np.float32)
        q_noise = rng.standard_normal((batch_size, flow_steps, future_steps, 7)).astype(np.float32)
        q_flow = q_target[:, None] + 0.2 * q_noise
        fk_xyz = np.einsum("bsfd,dk->bsfk", q_flow, w_q2x).astype(np.float32)[..., None, :]
        return {
            "text_conditioned": {
                "vis": {
                    "joints_flow_steps": pred_steps,
                    "joints_target_ft": target,
                    "q_flow_steps": q_flow,
                    "fk_xyz_flow_steps": fk_xyz,
                }
            }
        }

    return _eval_step


def _save_wandb_video_artifact(wb_vid, out_path: Path, fps: int) -> None:
    if hasattr(wb_vid, "data"):
        frames = wb_vid.data.transpose(0, 2, 3, 1)
        imageio.mimwrite(out_path, frames, fps=fps)
        return
    src = getattr(wb_vid, "_path", None)
    if src is None:
        raise ValueError("Unsupported wandb.Video object; no .data or ._path")
    shutil.copyfile(src, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo denoise plots from FlowVisCallback.")
    parser.add_argument("--wait-ms", type=int, default=1, help="OpenCV wait per refresh in ms.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--future-steps", type=int, default=6)
    parser.add_argument("--joints", type=int, default=21)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--num-val-batches", type=int, default=4, help="How many denoise plots to generate/show.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out",
        help="Directory to save denoise plot PNGs.",
    )
    args = parser.parse_args()

    wandb.init(mode="disabled")

    cb = object.__new__(FlowVisCallback)
    cb.fps = args.fps
    cb.max_videos = args.num_val_batches
    cb.num_val_batches = args.num_val_batches
    cb.camera_intrinsics = _DEFAULT_K.copy()
    cb.camera_R = np.eye(3, dtype=np.float32)
    cb.camera_t = np.zeros(3, dtype=np.float32)
    cb.use_rerun = False
    cb.rerun_spawn = False
    cb.rerun_path = None
    cb.ros_to_opencv = False
    cb.enable_denoise_plots = True
    cb.enable_xyz_image_flow = False
    cb.enable_joint_xyz_pca_flow = False
    cb.show_denoise_plot_window = True
    cb.denoise_plot_window_wait_ms = args.wait_ms
    cb.denoise_pred_keys = ("joints_flow_steps",)
    cb.denoise_target_keys = ("joints_target_ft",)
    cb.flow_q_keys = ("q_flow_steps",)
    cb.flow_xyz_keys = ("fk_xyz_flow_steps", "joints_flow_steps")
    cb.enable_part_a = True
    cb.enable_part_b = True
    cb.enable_part_c = True
    cb._rerun_initialized = False
    cb._fk_fn = None
    cb._robot = None
    cb.val_iterators = {"dummy": _dummy_iter()}
    cb.eval_step = _make_eval_step(
        seed=args.seed,
        batch_size=args.batch_size,
        flow_steps=args.flow_steps,
        future_steps=args.future_steps,
        joints=args.joints,
    )

    out = cb(train_state=None, step=0)
    print("Generated keys:", sorted(out.keys()))
    scalar_key = "flow_vis/dummy/denoise_last_l2_to_target"
    if scalar_key in out:
        print(f"{scalar_key}={out[scalar_key]:.6f}")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    denoise_key = "flow_vis/dummy/denoise_plots"
    saved = 0
    for idx, wb_img in enumerate(out.get(denoise_key, [])):
        png_path = out_dir / f"denoise_plot_{idx:03d}.png"
        wb_img.image.save(png_path)
        saved += 1
    points_key = "flow_vis/dummy/denoise_points_3d"
    saved_vid = 0
    for idx, wb_vid in enumerate(out.get(points_key, [])):
        gif_path = out_dir / f"denoise_points_3d_{idx:03d}.gif"
        _save_wandb_video_artifact(wb_vid, gif_path, fps=args.fps)
        saved_vid += 1
    overlay_key = "flow_vis/dummy/xyz_overlay"
    saved_overlay = 0
    for idx, wb_vid in enumerate(out.get(overlay_key, [])):
        mp4_path = out_dir / f"xyz_overlay_{idx:03d}.mp4"
        _save_wandb_video_artifact(wb_vid, mp4_path, fps=args.fps)
        saved_overlay += 1
    pca_key = "flow_vis/dummy/joint_fk_pca"
    saved_pca = 0
    for idx, wb_img in enumerate(out.get(pca_key, [])):
        png_path = out_dir / f"joint_fk_pca_{idx:03d}.png"
        wb_img.image.save(png_path)
        saved_pca += 1
    robot_key = "flow_vis/dummy/robot_flow"
    saved_robot = 0
    for idx, wb_vid in enumerate(out.get(robot_key, [])):
        mp4_path = out_dir / f"robot_flow_{idx:03d}.mp4"
        _save_wandb_video_artifact(wb_vid, mp4_path, fps=args.fps)
        saved_robot += 1
    if scalar_key in out:
        (out_dir / "denoise_metrics.txt").write_text(
            f"{scalar_key}={out[scalar_key]:.6f}\n",
            encoding="utf-8",
        )
    print(
        f"Saved denoise plots={saved}, denoise_3d_gifs={saved_vid}, "
        f"xyz_overlay_gifs={saved_overlay}, joint_fk_pca_plots={saved_pca}, "
        f"robot_flow_gifs={saved_robot} to: {out_dir}"
    )

    if cb.show_denoise_plot_window:
        import cv2

        print("Press any key in an OpenCV window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
