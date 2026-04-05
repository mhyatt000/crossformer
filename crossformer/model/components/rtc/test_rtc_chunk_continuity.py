"""RTC chunk continuity and jitter visualization tests.

Tests Matt's requests:
  - (2) Does RTC commit to one plan rather than switching (jitter)?
  - (4) Produce a figure like the one from the RTC website.

Both work with random XFlowHead weights — no checkpoint needed.
When a real checkpoint is available, swap `bound_head` for the loaded model.

Run with:
    pytest crossformer/model/components/rtc/test_rtc_chunk_continuity.py -v

To generate the figure only:
    pytest crossformer/model/components/rtc/test_rtc_chunk_continuity.py \
        -v -k "figure"
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from crossformer.model.components.rtc.rtc_algorithm import guided_inference

# ---------------------------------------------------------------------------
# Pull constants from conftest
# ---------------------------------------------------------------------------

# bununla değiştir:
H     = 6
max_A = 4
B     = 1
W     = 1


# Paper Table 4
FLOW_STEPS = 5
BETA       = 5.0
D          = 1   # inference delay
S          = 3   # execution horizon

SEEDS      = [0, 1, 2, 3, 4]
FIG_DIR    = Path("figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_guided(bound_head, obs, A_prev, seed: int) -> np.ndarray:
    """Run guided_inference for one seed → (H, max_A)."""
    out = guided_inference(
        bound_head,
        {**obs, "rng": jax.random.PRNGKey(seed)},
        A_prev,
        d=D, s=S,
        flow_steps=FLOW_STEPS,
        beta=BETA,
    )
    return np.array(out[0, 0])  # (H, max_A)


def _run_naive(bound_head, obs, seed: int) -> np.ndarray:
    """Naive async: no guidance, fresh chunk each time → (H, max_A)."""
    out = guided_inference(
        bound_head,
        {**obs, "rng": jax.random.PRNGKey(seed)},
        jnp.zeros((H, max_A)),   # A_prev ignored: d=0, s=H → mask all zero
        d=0, s=H,
        flow_steps=FLOW_STEPS,
        beta=BETA,
    )
    return np.array(out[0, 0])


def _boundary_jump(chunk_a: np.ndarray, chunk_b: np.ndarray, s: int) -> float:
    """L2 distance between last executed action and first action of next chunk.

    chunk_a[s-1] is the last action executed from chunk_a.
    chunk_b[0]   is the first action of the new chunk.
    """
    return float(np.linalg.norm(chunk_a[s - 1] - chunk_b[0]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJitter:
    """Matt request (2): does RTC commit to one plan vs. switching?

    We measure the boundary jump between consecutive chunks:
        RTC    : guided_inference uses soft mask → A_new[0] close to A_prev[s-1]
        Naive  : no guidance     → A_new[0] unrelated to A_prev[s-1]

    Assert: mean RTC jump < mean naive jump across seeds.
    """

    def test_rtc_reduces_boundary_jump(self, bound_head, obs):
        A_prev = jnp.ones((H, max_A), dtype=jnp.float32) * 2.0
        A_prev_np = np.array(A_prev)

        rtc_jumps   = []
        naive_jumps = []

        for seed in SEEDS:
            A_rtc   = _run_guided(bound_head, obs, A_prev, seed)
            A_naive = _run_naive(bound_head, obs, seed)

            rtc_jumps.append(_boundary_jump(A_prev_np, A_rtc,   s=S))
            naive_jumps.append(_boundary_jump(A_prev_np, A_naive, s=S))

        mean_rtc   = np.mean(rtc_jumps)
        mean_naive = np.mean(naive_jumps)

        assert mean_rtc < mean_naive, (
            f"RTC mean boundary jump ({mean_rtc:.4f}) >= "
            f"naive mean boundary jump ({mean_naive:.4f}) — "
            f"RTC is not reducing jitter"
        )

    def test_rtc_frozen_region_no_jump(self, bound_head, obs):
        """Frozen region [0:d] must stay close to A_prev — this is the hard commit."""
        A_prev    = jnp.ones((H, max_A), dtype=jnp.float32) * 2.0
        A_prev_np = np.array(A_prev)

        for seed in SEEDS:
            A_rtc = _run_guided(bound_head, obs, A_prev, seed)
            frozen_jump = float(np.linalg.norm(A_rtc[:D] - A_prev_np[:D]))
            naive_A = _run_naive(bound_head, obs, seed)
            naive_jump = float(np.linalg.norm(naive_A[:D] - A_prev_np[:D]))

            assert frozen_jump < naive_jump, (
                f"seed={seed}: RTC frozen jump ({frozen_jump:.4f}) >= "
                f"naive jump ({naive_jump:.4f})"
            )


class TestChunkFigure:
    """Matt request (4): produce a figure like the RTC website.

    Shows consecutive predicted chunks over time — RTC commits to one
    trajectory while naive async scatters across strategies.

    Output: figures/rtc_chunk_figure.png
    """

    def test_generate_figure(self, bound_head, obs):
        FIG_DIR.mkdir(exist_ok=True)

        # Simulate 3 consecutive inference calls, DOF index 0
        DOF_IDX = 0
        A_prev = jnp.ones((H, max_A), dtype=jnp.float32) * 2.0

        rtc_chunks   = []
        naive_chunks = []

        for seed in SEEDS[:3]:
            rtc_chunks.append(_run_guided(bound_head, obs, A_prev, seed)[:, DOF_IDX])
            naive_chunks.append(_run_naive(bound_head, obs, seed)[:, DOF_IDX])
            # next A_prev is the output of this RTC call
            A_prev = jnp.array(_run_guided(bound_head, obs, A_prev, seed))

        # -- plot --
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig.suptitle("RTC vs Naive Async — Consecutive Chunks (DOF 0)", fontsize=13)

        colors = ["#2d6a4f", "#74c69d", "#b7e4c7"]  # dark → light green (paper style)

        for ax, chunks, title in zip(axes, [rtc_chunks, naive_chunks], ["RTC (ours)", "Naive Async"]):
            for i, (chunk, color) in enumerate(zip(chunks, colors)):
                x = np.arange(len(chunk))
                alpha = 1.0 - i * 0.2
                ax.plot(x, chunk, color=color, alpha=alpha, linewidth=2,
                        label=f"chunk {i+1}")
                ax.scatter(x, chunk, color=color, alpha=alpha, s=40, zorder=3)

            # mark frozen region
            ax.axvspan(0, D - 0.5, color="#f4a261", alpha=0.15, label=f"frozen (d={D})")
            # mark fresh region
            ax.axvspan(H - S - 0.5, H - 0.5, color="#e9c46a", alpha=0.15, label=f"fresh (s={S})")

            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Timestep within chunk")
            ax.set_ylabel("Action value (DOF 0)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = FIG_DIR / "rtc_chunk_figure.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        assert out_path.exists(), f"Figure not saved to {out_path}"
        print(f"\nFigure saved → {out_path}")
