from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import jax
import jax.numpy as jnp
import jaxlie
import pyroki as pk
import yourdfpy
from xgym import calibrate
from xgym.calibrate.urdf.robot import urdf

from crossformer.data.grain.datasets import _DecodedArrayRecord


# -------------------------
# utilities
# -------------------------
def _get_by_path(obj: dict, path: str):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _fit_pca_2d(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    x: [M, D]
    returns:
      z2: [M, 2]
      evr: [2] explained variance ratio
    """
    x = np.asarray(x, dtype=np.float32)
    xz = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(xz)
    return z2, pca.explained_variance_ratio_


def _ensure_snd(x: np.ndarray, name: str) -> np.ndarray:
    """
    Normalize to [S, N, D]
      [S, D] -> [S, 1, D]
      [S, N, D] -> as-is
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim != 3:
        raise ValueError(f"{name} must be [S,D] or [S,N,D], got {x.shape}")
    return x


def _flatten_k3ds_xyz(k3ds: np.ndarray) -> np.ndarray:
    """
    k3ds:
      [S, J, 4] -> [S, J*3]
      [S, N, J, 4] -> [S, N, J*3]
      [S, J, 3] -> [S, J*3]
      [S, N, J, 3] -> [S, N, J*3]
    """
    k = np.asarray(k3ds, dtype=np.float32)
    if k.ndim == 3 and k.shape[-1] in (3, 4):
        return k[..., :3].reshape(k.shape[0], -1)
    if k.ndim == 4 and k.shape[-1] in (3, 4):
        return k[..., :3].reshape(k.shape[0], k.shape[1], -1)
    raise ValueError(f"Unsupported k3ds shape {k.shape}")


# -------------------------
# robot FK
# -------------------------
def make_robot() -> pk.Robot:
    urdf_model = yourdfpy.URDF.load(
        urdf, mesh_dir=calibrate.urdf.robot.DNAME / "assets"
    )
    return pk.Robot.from_urdf(urdf_model)


def get_fk_fn_all_links(robot: pk.Robot, pad_gripper: bool = True):
    """
    Input q:
      [S, Dq] or [S, N, Dq]
    Output:
      [S, L*3] or [S, N, L*3]
    where L = number of robot links.
    """
    def fk_fn(q: jax.Array) -> jax.Array:
        pads = [(0, 0)] * (q.ndim - 1) + [(0, 1)]  # add gripper dim if needed
        q_pad = jnp.pad(q, pads, constant_values=0.0) if pad_gripper else q
        t = jaxlie.SE3(robot.forward_kinematics(q_pad)).translation()  # [..., L, 3]
        return t.reshape(*t.shape[:-2], -1)  # [..., L*3]

    return fk_fn


# -------------------------
# plotting / animation
# -------------------------
def _axis_limits(z2: np.ndarray, pad=0.10):
    mn, mx = z2.min(0), z2.max(0)
    d = np.maximum(mx - mn, 1e-6)
    return (
        mn[0] - pad * d[0],
        mx[0] + pad * d[0],
        mn[1] - pad * d[1],
        mx[1] + pad * d[1],
    )


def _plot_two_panel_video(
    q_flow: np.ndarray,      # [S,N,Dq] (N may be 1)
    x_flow: np.ndarray,      # [S,N,Dx] (N may be 1)
    out_png: Path,
    out_mp4: Path,
    fps: int = 8,
):
    S, N, Dq = q_flow.shape
    S2, N2, Dx = x_flow.shape
    if (S, N) != (S2, N2):
        raise ValueError(f"shape mismatch q={q_flow.shape}, x={x_flow.shape}")

    # PCA fit across all step+sample points so axes are fixed over time
    q_all = q_flow.reshape(S * N, Dq)
    x_all = x_flow.reshape(S * N, Dx)

    q2_all, q_evr = _fit_pca_2d(q_all)  # [S*N,2]
    x2_all, x_evr = _fit_pca_2d(x_all)

    q2 = q2_all.reshape(S, N, 2)
    x2 = x2_all.reshape(S, N, 2)

    qlim = _axis_limits(q2_all)
    xlim = _axis_limits(x2_all)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
    for ax in (ax0, ax1):
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    ax0.set_title(f"Joint PCA  EVR=({q_evr[0]:.2f},{q_evr[1]:.2f})")
    ax1.set_title(f"FK XYZ PCA EVR=({x_evr[0]:.2f},{x_evr[1]:.2f})")

    ax0.set_xlim(qlim[0], qlim[1]); ax0.set_ylim(qlim[2], qlim[3])
    ax1.set_xlim(xlim[0], xlim[1]); ax1.set_ylim(xlim[2], xlim[3])

    # artists
    # cloud points (current step)
    q_sc = ax0.scatter([], [], c="blue", s=20, alpha=0.25)
    x_sc = ax1.scatter([], [], c="red", s=20, alpha=0.25)
    # centroid
    q_ctr = ax0.scatter([], [], c="navy", s=65, marker="x")
    x_ctr = ax1.scatter([], [], c="darkred", s=65, marker="x")
    # trajectory of centroids across steps
    q_line, = ax0.plot([], [], color="blue", lw=1.6, alpha=0.85)
    x_line, = ax1.plot([], [], color="red", lw=1.6, alpha=0.85)

    title = fig.suptitle("")

    def update(k: int):
        qk = q2[k]  # [N,2]
        xk = x2[k]  # [N,2]

        q_sc.set_offsets(qk)
        x_sc.set_offsets(xk)

        qmu = qk.mean(0)
        xmu = xk.mean(0)
        q_ctr.set_offsets(qmu[None, :])
        x_ctr.set_offsets(xmu[None, :])

        q_hist = q2[: k + 1].mean(1)  # [k+1,2]
        x_hist = x2[: k + 1].mean(1)
        q_line.set_data(q_hist[:, 0], q_hist[:, 1])
        x_line.set_data(x_hist[:, 0], x_hist[:, 1])

        mode = "cloud" if N > 1 else "trajectory"
        title.set_text(f"{mode} step {k + 1}/{S}  (N={N})")
        return q_sc, x_sc, q_ctr, x_ctr, q_line, x_line, title

    update(0)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    print(f"[ok] wrote {out_png}")

    anim = FuncAnimation(fig, update, frames=S, interval=1000 / fps, blit=False)
    anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print(f"[ok] wrote {out_mp4}")

    # simple numeric diagnostics
    q_spread = np.linalg.norm(q2 - q2.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)  # [S]
    x_spread = np.linalg.norm(x2 - x2.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)  # [S]
    print("[diag] joint spread first/last:", float(q_spread[0]), float(q_spread[-1]))
    print("[diag] fk    spread first/last:", float(x_spread[0]), float(x_spread[-1]))


# -------------------------
# data loading
# -------------------------
def _load_q_from_record(step: dict, q_path: str, ft_index: int) -> np.ndarray:
    """
    q_path examples:
      action.q          -> [S,7] or [S,ft,7]
      observation.q     -> [S,7] etc
      action.k3ds       -> [S,J,4] or [S,N,J,4] (xyz flattened)
    Returns:
      q_flow [S,D] or [S,N,D]
    """
    q_raw = _get_by_path(step, q_path)
    if q_raw is None:
        raise KeyError(f"Missing `{q_path}` in record.")

    q = np.asarray(q_raw, dtype=np.float32)

    # If path points to k3ds, use xyz flattened as joint representation.
    if q.ndim in (3, 4) and q.shape[-1] in (3, 4):
        return _flatten_k3ds_xyz(q)

    # Otherwise assume q angles
    if q.ndim == 2:          # [S,D]
        return q
    if q.ndim == 3:          # [S,ft,D] or [S,N,D]
        # if middle dim small and user gave ft index, treat as future-time dim
        if q.shape[1] > ft_index and q.shape[1] <= 64:
            return q[:, ft_index, :]
        return q

    raise ValueError(f"Unsupported q shape {q.shape} from path {q_path}")


def _load_q_from_npy(q_npy: Path, key: str, ft_index: int) -> np.ndarray:
    p = q_npy.expanduser().resolve()
    arr = np.load(p)[key] if p.suffix == ".npz" else np.load(p)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim in (3, 4) and arr.shape[-1] in (3, 4):
        return _flatten_k3ds_xyz(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[1] > ft_index and arr.shape[1] <= 64:
            return arr[:, ft_index, :]
        return arr

    raise ValueError(f"Unsupported q_npy shape {arr.shape}")


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Two-panel PCA convergence video: Joint (left) vs FK XYZ (right), supports trajectory and cloud."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("~/crossformer_data").expanduser())
    parser.add_argument("--record-idx", type=int, default=0)

    parser.add_argument(
        "--q-path",
        type=str,
        default="action.q",
        help="Path for joint flow in record. Can be q (angles) or k3ds.",
    )
    parser.add_argument("--q-npy", type=Path, default=None, help="Optional .npy/.npz for q flow (overrides --q-path).")
    parser.add_argument("--q-npy-key", type=str, default="q_flow_steps")
    parser.add_argument("--ft-index", type=int, default=0)

    parser.add_argument("--pad-gripper", action="store_true", default=True)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("~/repos/crossformer/tests/out").expanduser())
    args = parser.parse_args()

    shards = sorted(args.data_dir.glob("**/*.arrayrecord"))
    records = _DecodedArrayRecord(shards)
    print(f"[info] records={len(records)} shards={len(shards)}")
    if len(records) == 0:
        raise RuntimeError("No records found.")
    step = records[args.record_idx]

    if args.q_npy is not None:
        q_flow = _load_q_from_npy(args.q_npy, args.q_npy_key, args.ft_index)
        print(f"[info] loaded q from npy: {args.q_npy}")
    else:
        q_flow = _load_q_from_record(step, args.q_path, args.ft_index)
        print(f"[info] loaded q from record path: {args.q_path}")

    q_flow = _ensure_snd(q_flow, "q_flow")   # [S,N,Dq]
    S, N, Dq = q_flow.shape
    print(f"[info] q_flow shape={q_flow.shape}")

    # FK from q
    robot = make_robot()
    fk_fn = get_fk_fn_all_links(robot, pad_gripper=args.pad_gripper)

    q_jax = jnp.asarray(q_flow)
    x_flow = np.asarray(fk_fn(q_jax), dtype=np.float32)  # [S,N,Dx] or [S,Dx]
    x_flow = _ensure_snd(x_flow, "x_flow")
    print(f"[info] x_flow shape={x_flow.shape}")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "pca_convergence_step0.png"
    out_mp4 = out_dir / "pca_convergence.mp4"

    _plot_two_panel_video(
        q_flow=q_flow,
        x_flow=x_flow,
        out_png=out_png,
        out_mp4=out_mp4,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
