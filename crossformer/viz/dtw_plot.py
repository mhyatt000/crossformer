from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def render_dtw_alignment_figure(
    seq_a: np.ndarray, seq_b: np.ndarray, path: np.ndarray, title_a="Human (Series A)", title_b="Robot (Series B)"
) -> np.ndarray:
    """
    Renders the 3-panel DTW alignment plot and returns it as an RGB image array.

    Args:
        seq_a: 1D np.array (human joint traj.)
        seq_b: 1D np.array (robot joing traj.)
        path: np.array of shape (K, 2) containing warping path indices.

    Returns:
        rgb_frame: np.ndarray of shape (H, W, 3) of type uint8
    """
    fig = plt.figure(figsize=(14, 8), dpi=120)

    # Top Left: Original Time Series
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(seq_a, label=title_a, color="blue")
    ax1.plot(seq_b, label=title_b, color="orange", linestyle="--")
    ax1.set_title("Original Time Series")
    ax1.legend()

    # Top Right: The Warping Path Matrix
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(path[:, 0], path[:, 1], marker="o", color="green", markersize=4)
    ax2.set_title("Shortest Path (Warping Path)")
    ax2.set_xlabel(f"{title_a} Index")
    ax2.set_ylabel(f"{title_b} Index")
    ax2.grid(True)

    # Bottom: Point-to-Point Alignment
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(seq_a, marker="o", color="blue", label=title_a)
    ax3.plot(seq_b, marker="x", color="orange", linestyle="--", label=title_b)

    for i, j in path:
        ax3.plot([i, j], [seq_a[i], seq_b[j]], color="gray", alpha=0.4, linewidth=1)

    ax3.set_title("Point-to-Point Comparison After DTW Alignment")
    ax3.legend()

    fig.tight_layout()

    # Draw to canvas and extract as numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb_frame = buf[:, :, :3].copy()

    plt.close(fig)
    return rgb_frame
