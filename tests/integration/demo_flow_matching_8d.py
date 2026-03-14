from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fit_pca_basis_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    x0 = x - mu
    _, _, vh = np.linalg.svd(x0, full_matrices=False)
    basis = vh[:2].T
    return mu[0], basis


def _project_pca_2d(x: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - mu) @ basis


def _make_known_distribution(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    mean = np.linspace(-1.25, 1.25, dim, dtype=np.float32)
    a = rng.standard_normal((dim, dim)).astype(np.float32)
    cov = (a @ a.T) / dim + 0.2 * np.eye(dim, dtype=np.float32)
    return rng.multivariate_normal(mean=mean, cov=cov, size=n).astype(np.float32)


def _render_frame(
    x_t: np.ndarray,
    x_target: np.ndarray,
    t: float,
    dim_colors: np.ndarray,
    pca_mu: np.ndarray,
    pca_basis: np.ndarray,
    jitter: np.ndarray,
) -> np.ndarray:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    # Left: per-dimension values, colored by dimension index
    xs = np.arange(x_t.shape[1], dtype=np.float32)
    for d in range(x_t.shape[1]):
        ax0.scatter(
            np.full(x_t.shape[0], xs[d]) + jitter[:, d],
            x_t[:, d],
            s=5,
            alpha=0.35,
            color=dim_colors[d],
            label=f"dim {d}",
        )
    ax0.set_title(f"8D refinement values (t={t:.2f})")
    ax0.set_xlabel("dimension index")
    ax0.set_ylabel("value")
    ax0.set_xlim(-0.6, x_t.shape[1] - 0.4)
    ax0.grid(alpha=0.2)
    if t == 0.0:
        ax0.legend(ncol=2, fontsize=7, loc="upper left")

    # Right: PCA projection of current 8D features
    z_t = _project_pca_2d(x_t, pca_mu, pca_basis)
    z_target = _project_pca_2d(x_target, pca_mu, pca_basis)
    ax1.scatter(z_target[:, 0], z_target[:, 1], s=6, alpha=0.15, color="gray", label="target")
    ax1.scatter(z_t[:, 0], z_t[:, 1], s=8, alpha=0.65, color="#1f77b4", label="current")
    ax1.set_title(f"PCA(8D→2D) refinement (t={t:.2f})")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.grid(alpha=0.2)
    if t == 0.0:
        ax1.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.canvas.draw()
    h, w = fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0]
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-matching refinement demo in 8D with PCA video.")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--num-points", type=int, default=600)
    parser.add_argument("--num-steps", type=int, default=80)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "out" / "flow_match_8d.mp4",
        help="Output mp4 path.",
    )
    args = parser.parse_args()
    if args.dim != 8:
        raise ValueError("This demo is fixed to dim=8 per request.")

    rng = np.random.default_rng(args.seed)
    x0 = rng.standard_normal((args.num_points, args.dim)).astype(np.float32)
    x1 = _make_known_distribution(rng, n=args.num_points, dim=args.dim)

    pca_mu, pca_basis = _fit_pca_basis_2d(np.concatenate([x0, x1], axis=0))
    colors = plt.cm.tab10(np.linspace(0, 1, args.dim))
    jitter = 0.18 * (rng.random((args.num_points, args.dim), dtype=np.float32) - 0.5)

    ts = np.linspace(0.0, 1.0, args.num_steps, dtype=np.float32)
    frames = []
    for t in ts:
        x_t = (1.0 - t) * x0 + t * x1
        frames.append(
            _render_frame(
                x_t=x_t,
                x_target=x1,
                t=float(t),
                dim_colors=colors,
                pca_mu=pca_mu,
                pca_basis=pca_basis,
                jitter=jitter,
            )
        )

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=args.fps, quality=8)
    print(f"Saved video: {out_path}")


if __name__ == "__main__":
    main()
