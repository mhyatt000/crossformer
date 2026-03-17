from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path

import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import yourdfpy


def _ensure_snd(x: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize to [S, N, D].
      [S, D]    -> [S, 1, D]
      [S, N, D] -> as-is
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must be [S,D] or [S,N,D], got {x.shape}")


def _rotvec_from_matrix(r: np.ndarray) -> np.ndarray:
    tr = np.trace(r)
    c = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(c))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float32)

    v = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=np.float64,
    )
    s = np.linalg.norm(v)
    if s < 1e-8:
        w, vecs = np.linalg.eigh(r)
        axis = vecs[:, np.argmax(w)]
    else:
        axis = v / s
    return (theta * axis).astype(np.float32)


def _urdf_paths() -> tuple[Path, Path]:
    spec = find_spec("xgym")
    if spec is None or spec.origin is None:
        raise RuntimeError("xgym is required for FK PCA visualization")
    root = Path(spec.origin).resolve().parent
    urdf_dir = root / "calibrate" / "urdf"
    return urdf_dir / "xarm7_standalone.urdf", urdf_dir / "assets"


def _make_fk_fn():
    urdf_path, mesh_dir = _urdf_paths()
    robot = yourdfpy.URDF.load(urdf_path, mesh_dir=mesh_dir)

    def fk_fn(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        q: [N,7]
        returns:
          xyz: [N,3]
          rotvec: [N,3]
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1] < 7:
            raise ValueError(f"Expected q dim >=7, got {q.shape}")
        q7 = q[:, :7]
        q8 = np.pad(q7, ((0, 0), (0, 1)))  # add gripper dummy
        xyz = np.empty((q8.shape[0], 3), dtype=np.float32)
        rot = np.empty((q8.shape[0], 3), dtype=np.float32)
        for i, qi in enumerate(q8):
            robot.update_cfg(qi)
            tf = robot.get_transform("link_eef")
            xyz[i] = tf[:3, 3]
            rot[i] = _rotvec_from_matrix(tf[:3, :3])
        return xyz, rot

    return fk_fn


def _fit_pca(*xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate(xs, axis=0)
    mean = x.mean(axis=0, keepdims=True)
    x0 = x - mean
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:2].T
    return mean.astype(np.float32), basis.astype(np.float32)


def _project_pca(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mean) @ basis


def _frame_image(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def _plot_panel(ax: plt.Axes, z: np.ndarray, t_text: str, lim: float, color: str, title: str):
    ax.scatter(z[:, 0], z[:, 1], s=12, alpha=0.35, linewidths=0.0, color=color)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("pc 1")
    ax.set_ylabel("pc 2")
    ax.set_title(f"{title}   {t_text}")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")


def _safe_lim(z0: np.ndarray, z1: np.ndarray) -> float:
    v = np.abs(np.concatenate([z0, z1], axis=0)).max()
    return float(max(1e-3, 1.05 * v))


def _plot_two_panel_video(
    q_flow: np.ndarray,  # [S,D] or [S,N,D]
    x_flow: np.ndarray | None,  # ignored (kept for callback compatibility)
    out_png: str | Path,
    out_mp4: str | Path,
    fps: int = 8,
):
    """
    Compatibility entrypoint used by callback.
    Now produces 4 panels:
      1) Joint PCA
      2) FK XYZ PCA
      3) FK Rot PCA
      4) FK Pose PCA (xyz+rot)
    """
    del x_flow  # no longer needed; computed from q via FK

    q_snd = _ensure_snd(q_flow, "q_flow")  # [S,N,D]
    S, N, D = q_snd.shape
    if D < 7:
        raise ValueError(f"Need q dim >= 7, got {q_snd.shape}")

    fk_fn = _make_fk_fn()

    # Endpoints to define stable PCA bases/limits
    q0 = q_snd[0][:, :7]  # [N,7]
    q1 = q_snd[-1][:, :7]  # [N,7]
    x0, r0 = fk_fn(q0)  # [N,3], [N,3]
    x1, r1 = fk_fn(q1)
    p0 = np.concatenate([x0, r0], axis=1)  # [N,6]
    p1 = np.concatenate([x1, r1], axis=1)

    q_mean, q_basis = _fit_pca(q0, q1)
    x_mean, x_basis = _fit_pca(x0, x1)
    r_mean, r_basis = _fit_pca(r0, r1)
    p_mean, p_basis = _fit_pca(p0, p1)

    q_lim = _safe_lim(_project_pca(q0, q_mean, q_basis), _project_pca(q1, q_mean, q_basis))
    x_lim = _safe_lim(_project_pca(x0, x_mean, x_basis), _project_pca(x1, x_mean, x_basis))
    r_lim = _safe_lim(_project_pca(r0, r_mean, r_basis), _project_pca(r1, r_mean, r_basis))
    p_lim = _safe_lim(_project_pca(p0, p_mean, p_basis), _project_pca(p1, p_mean, p_basis))

    out_png = Path(out_png)
    out_mp4 = Path(out_mp4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5), dpi=128)
    frames: list[np.ndarray] = []

    for s in range(S):
        for ax in axs:
            ax.clear()

        q_t = q_snd[s][:, :7]  # [N,7]
        x_t, r_t = fk_fn(q_t)  # [N,3], [N,3]
        p_t = np.concatenate([x_t, r_t], axis=1)

        t_text = f"step={s + 1}/{S} (N={N})"

        _plot_panel(axs[0], _project_pca(q_t, q_mean, q_basis), t_text, q_lim, "blue", "Joint PCA")
        _plot_panel(axs[1], _project_pca(x_t, x_mean, x_basis), t_text, x_lim, "red", "FK XYZ PCA")
        _plot_panel(axs[2], _project_pca(r_t, r_mean, r_basis), t_text, r_lim, "green", "FK Rot PCA")
        _plot_panel(axs[3], _project_pca(p_t, p_mean, p_basis), t_text, p_lim, "purple", "FK Pose PCA")

        fig.tight_layout()
        frame = _frame_image(fig)
        frames.append(frame)

    plt.close(fig)

    imageio.imwrite(out_png, frames[0])
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264") as writer:
        for fr in frames:
            writer.append_data(fr)

    return {
        "out_png": str(out_png),
        "out_mp4": str(out_mp4),
        "num_steps": S,
        "num_points": N,
    }


def _extract_q(step: dict) -> np.ndarray:
    q_raw = None
    k3_raw = None

    action = step.get("action")
    if isinstance(action, dict):
        q_raw = action.get("q")
        k3_raw = action.get("k3ds")
    if q_raw is None:
        q_raw = step.get("q")
    if k3_raw is None:
        k3_raw = step.get("k3ds")

    if q_raw is not None:
        q = np.asarray(q_raw, dtype=np.float32)
        if q.ndim == 3:
            q = q[:, 0, :]
        if q.ndim != 2:
            raise ValueError(f"Expected q with ndim 2/3, got {q.shape}")
        return q
    if k3_raw is not None:
        k3 = np.asarray(k3_raw, dtype=np.float32)  # [S,J,3/4]
        if k3.ndim != 3:
            raise ValueError(f"Expected k3ds with ndim 3, got {k3.shape}")
        return k3[:, :, :3].reshape(k3.shape[0], -1)  # [S,D]

    keys = sorted(step.keys()) if isinstance(step, dict) else []
    raise KeyError(f"Could not find q/k3ds in record. Available keys: {keys}")


def _load_q_from_dataset(
    path: Path,
    traj_index: int,
    max_steps: int | None,
    sample_mode: str,
    num_samples: int,
) -> np.ndarray:
    from crossformer.data.grain.datasets import _DecodedArrayRecord

    shards = sorted(path.expanduser().glob("**/*.arrayrecord"))
    if not shards:
        raise FileNotFoundError(f"No ArrayRecord shards found in {path}")

    records = _DecodedArrayRecord(shards)
    if traj_index < 0 or traj_index >= len(records):
        raise IndexError(f"traj_index {traj_index} out of range [0, {len(records) - 1}]")

    if num_samples < 1:
        raise ValueError(f"num_samples must be >=1, got {num_samples}")

    q = _extract_q(records[traj_index])
    if max_steps is not None:
        q = q[:max_steps]

    if sample_mode == "single":
        out = q
    elif sample_mode == "window":
        n = int(num_samples)
        offs = np.arange(n, dtype=np.int32) - (n // 2)
        S = q.shape[0]
        idx = np.arange(S, dtype=np.int32)[:, None] + offs[None, :]
        idx = np.clip(idx, 0, S - 1)
        out = q[idx]  # [S,N,D]
    elif sample_mode == "episodes":
        if traj_index + num_samples > len(records):
            raise IndexError(
                f"Need {num_samples} records from traj_index={traj_index}, "
                f"but only {len(records) - traj_index} available"
            )
        qs = []
        for i in range(traj_index, traj_index + num_samples):
            qi = _extract_q(records[i])
            if max_steps is not None:
                qi = qi[:max_steps]
            qs.append(qi)
        s = min(x.shape[0] for x in qs)
        d = qs[0].shape[1]
        if any(x.shape[1] != d for x in qs):
            shapes = [x.shape for x in qs]
            raise ValueError(f"Mismatched q dims across episodes: {shapes}")
        out = np.stack([x[:s] for x in qs], axis=1)  # [S,N,D]
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    if out.shape[0] < 2:
        raise ValueError(f"Need at least 2 steps for PCA video, got {out.shape[0]}")
    return out


def _cli() -> dict[str, str | int]:
    p = argparse.ArgumentParser(description="Render PCA flow video from ArrayRecord dataset.")
    p.add_argument("--dataset-dir", type=Path, required=True, help="Directory containing *.arrayrecord shards")
    p.add_argument("--traj-index", type=int, default=0, help="Record index to visualize")
    p.add_argument("--max-steps", type=int, default=80, help="Max trajectory steps to render")
    p.add_argument(
        "--sample-mode",
        type=str,
        choices=("single", "episodes", "window"),
        default="single",
        help="How to build N samples per step for PCA scatter",
    )
    p.add_argument("--num-samples", type=int, default=8, help="Number of samples (N) for episodes/window mode")
    p.add_argument("--fps", type=int, default=8, help="Output video fps")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/flow_viz_pca"), help="Output directory")
    args = p.parse_args()

    q = _load_q_from_dataset(
        args.dataset_dir,
        args.traj_index,
        args.max_steps,
        sample_mode=args.sample_mode,
        num_samples=args.num_samples,
    )
    q_flow = _ensure_snd(q, "q_flow")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    return _plot_two_panel_video(
        q_flow=q_flow,
        x_flow=None,
        out_png=args.out_dir / "pca_step0.png",
        out_mp4=args.out_dir / "pca.mp4",
        fps=args.fps,
    )


if __name__ == "__main__":
    print(_cli())
