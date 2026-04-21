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

from crossformer.model.components.rtc.rtc_algorithm import guided_inference, build_soft_mask

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
H     = 6
max_A = 4
B     = 1
W     = 1

# Paper Table 4
FLOW_STEPS = 5
BETA       = 5.0
D          = 1   # inference delay
S          = 3   # execution horizon

SEEDS   = [0, 1, 2, 3, 4]
FIG_DIR = Path("figures")


# ---------------------------------------------------------------------------
# Module-level helpers — accessible by all classes in this file
# ---------------------------------------------------------------------------

def _region_l2(A_new: np.ndarray, A_prev: np.ndarray, lo: int, hi: int) -> float:
    """Mean L2 distance between A_new[lo:hi] and A_prev[lo:hi].

    A_new : (H, max_A)
    A_prev: (H, max_A)
    """
    diff = A_new[lo:hi] - A_prev[lo:hi]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


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
    return np.array(out[0, 0])


def _run_naive(bound_head, obs, seed: int) -> np.ndarray:
    """Naive async: no guidance, fresh chunk each time → (H, max_A).

    d=0, s=H collapses the soft mask to all zeros — no A_prev influence.
    """
    out = guided_inference(
        bound_head,
        {**obs, "rng": jax.random.PRNGKey(seed)},
        jnp.zeros((H, max_A)),
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
        RTC   : guided_inference uses soft mask → A_new[0] close to A_prev[s-1]
        Naive : no guidance → A_new[0] unrelated to A_prev[s-1]

    Assert: mean RTC jump < mean naive jump across seeds.
    """

    def test_rtc_reduces_boundary_jump(self, bound_head, obs):
        A_prev    = jnp.ones((H, max_A), dtype=jnp.float32) * 2.0
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
            A_rtc       = _run_guided(bound_head, obs, A_prev, seed)
            frozen_jump = float(np.linalg.norm(A_rtc[:D] - A_prev_np[:D]))
            naive_A     = _run_naive(bound_head, obs, seed)
            naive_jump  = float(np.linalg.norm(naive_A[:D] - A_prev_np[:D]))

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

        DOF_IDX = 0
        A_prev  = jnp.ones((H, max_A), dtype=jnp.float32) * 2.0

        rtc_chunks   = []
        naive_chunks = []

        for seed in SEEDS[:3]:
            rtc_chunks.append(_run_guided(bound_head, obs, A_prev, seed)[:, DOF_IDX])
            naive_chunks.append(_run_naive(bound_head, obs, seed)[:, DOF_IDX])
            A_prev = jnp.array(_run_guided(bound_head, obs, A_prev, seed))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        fig.suptitle("RTC vs Naive Async — Consecutive Chunks (DOF 0)", fontsize=13)

        colors = ["#2d6a4f", "#74c69d", "#b7e4c7"]

        for ax, chunks, title in zip(
            axes, [rtc_chunks, naive_chunks], ["RTC (ours)", "Naive Async"]
        ):
            for i, (chunk, color) in enumerate(zip(chunks, colors)):
                x     = np.arange(len(chunk))
                alpha = 1.0 - i * 0.2
                ax.plot(x, chunk, color=color, alpha=alpha, linewidth=2,
                        label=f"chunk {i+1}")
                ax.scatter(x, chunk, color=color, alpha=alpha, s=40, zorder=3)

            ax.axvspan(0,       D - 0.5,   color="#f4a261", alpha=0.15,
                       label=f"frozen (d={D})")
            ax.axvspan(H-S-0.5, H - 0.5,   color="#e9c46a", alpha=0.15,
                       label=f"fresh (s={S})")
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


class TestDelayScaling:
    """Verifies soft mask correctness across d=0,1,2 inference delays.

    Tests two distinct properties:
      1. Policy-level: for d>0, frozen region error < fresh region error
         (this is a mathematical property of the guidance mechanism).
      2. Mask-level: build_soft_mask geometry is correct for each d
         (deterministic, no policy involved at all).

    No checkpoint needed — mask geometry is independent of learned weights.

    Constraint d <= s <= H-d with H=6, S=3:
        d=0: 0 <= 3 <= 6  OK
        d=1: 1 <= 3 <= 5  OK
        d=2: 2 <= 3 <= 4  OK
        d=3: intermediate region collapses (H-d == s), skipped
    """

    DELAY_VALUES = [0, 1, 2]

    @pytest.fixture(scope="class")
    def A_prev(self):
        return jnp.ones((H, max_A), dtype=jnp.float32) * 2.0

    def _run(self, bound_head, obs, A_prev, d: int, seed: int) -> np.ndarray:
        """Run guided_inference for one (d, seed) combination → (H, max_A)."""
        out = guided_inference(
            bound_head,
            {**obs, "rng": jax.random.PRNGKey(seed)},
            A_prev,
            d=d,
            s=S,
            flow_steps=FLOW_STEPS,
            beta=BETA,
        )
        return np.array(out[0, 0])

    # ------------------------------------------------------------------
    # Policy-level tests (use guided_inference, random weights OK)
    # ------------------------------------------------------------------

    def test_frozen_always_smaller_error_than_fresh(self, bound_head, obs, A_prev):
        """For every d > 0, frozen region error must be less than fresh region error.

        This holds regardless of policy weights because the soft mask
        applies W=1 to the frozen region and W=0 to the fresh region,
        making the guidance pull A_new toward A_prev only in frozen steps.
        """
        A_prev_np = np.array(A_prev)
        failures  = []

        for d in self.DELAY_VALUES:
            if d == 0:
                continue   # no frozen region when d=0
            for seed in SEEDS:
                A_new      = self._run(bound_head, obs, A_prev, d=d, seed=seed)
                frozen_err = _region_l2(A_new, A_prev_np, lo=0,   hi=d)
                fresh_err  = _region_l2(A_new, A_prev_np, lo=H-S, hi=H)
                if not (frozen_err < fresh_err):
                    failures.append(
                        f"d={d} seed={seed}: "
                        f"frozen={frozen_err:.4f} >= fresh={fresh_err:.4f}"
                    )

        assert not failures, "\n".join(failures)

    def test_d0_fresh_not_smaller_than_intermediate(self, bound_head, obs, A_prev):
        """With d=0 the entire non-fresh region gets soft guidance.

        intermediate [0:H-S] receives decaying guidance weights;
        fresh [H-S:H] receives W=0. So fresh_error >= intermediate_error.
        A small tolerance is allowed since guidance is approximate.
        """
        A_prev_np = np.array(A_prev)

        for seed in SEEDS:
            A_new            = self._run(bound_head, obs, A_prev, d=0, seed=seed)
            intermediate_err = _region_l2(A_new, A_prev_np, lo=0,   hi=H-S)
            fresh_err        = _region_l2(A_new, A_prev_np, lo=H-S, hi=H)
            assert fresh_err >= intermediate_err - 1e-3, (
                f"seed={seed} d=0: fresh ({fresh_err:.4f}) < "
                f"intermediate ({intermediate_err:.4f}) by more than tolerance"
            )

    def test_all_outputs_finite(self, bound_head, obs, A_prev):
        """All d values must produce finite outputs — no NaN or Inf."""
        for d in self.DELAY_VALUES:
            for seed in SEEDS:
                A_new = self._run(bound_head, obs, A_prev, d=d, seed=seed)
                assert np.all(np.isfinite(A_new)), (
                    f"d={d} seed={seed}: output contains NaN or Inf"
                )

    # ------------------------------------------------------------------
    # Mask-level tests (pure math, no policy involved)
    # ------------------------------------------------------------------

    def test_mask_frozen_count_equals_d(self):
        """build_soft_mask must set exactly d steps to W=1.0 (frozen region)."""
        for d in self.DELAY_VALUES:
            mask         = np.array(build_soft_mask(H, d=d, s=S))
            frozen_count = int(np.sum(mask == 1.0))
            assert frozen_count == d, (
                f"d={d}: expected {d} frozen steps (W=1), got {frozen_count}\n"
                f"mask={mask}"
            )

    def test_mask_fresh_region_is_zero(self):
        """build_soft_mask must set the last s steps to W=0.0 (fresh region)."""
        for d in self.DELAY_VALUES:
            mask         = np.array(build_soft_mask(H, d=d, s=S))
            fresh_region = mask[H-S:]
            assert np.all(fresh_region == 0.0), (
                f"d={d}: fresh region [H-S:H] contains non-zero weights: "
                f"{fresh_region}"
            )

    def test_mask_total_guidance_increases_with_d(self):
        """Total guidance (sum of W) must increase with d.

        Larger d → more W=1 steps → higher sum.
        Pure Equation 5 math, no policy involved.
        """
        sums = {d: float(np.sum(build_soft_mask(H, d=d, s=S)))
                for d in self.DELAY_VALUES}

        assert sums[0] < sums[1] < sums[2], (
            f"Mask sum should increase monotonically with d.\n"
            f"Got: d=0→{sums[0]:.4f}, d=1→{sums[1]:.4f}, d=2→{sums[2]:.4f}"
        )

    def test_mask_intermediate_weights_in_range(self):
        """Intermediate region weights must be strictly between 0 and 1."""
        for d in self.DELAY_VALUES:
            mask = np.array(build_soft_mask(H, d=d, s=S))
            # intermediate region: [d, H-S)
            if d >= H - S:
                continue   # no intermediate region for this d
            intermediate = mask[d:H-S]
            assert np.all(intermediate > 0.0) and np.all(intermediate < 1.0), (
                f"d={d}: intermediate region weights out of (0,1): {intermediate}"
            )
