"""Integration tests for RTC with a real (randomly initialized) XFlowHead.

These tests spin up a genuine XFlowHead, initialize its params with JAX,
bind it, and run it through guided_inference and the full RTC lifecycle.
No mocking — real JAX compilation on real (random) weights.

Run with:
    pytest crossformer/model/components/rtc/test_rtc_integration.py -v

Note: first run is slow (~30s) due to JAX JIT compilation.
      Subsequent runs benefit from the compilation cache.
"""

from __future__ import annotations

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.rtc.rtc_algorithm import (
    RTC,
    guided_inference,
)

# ---------------------------------------------------------------------------
# Constants matching dof.py / embody.py
# ---------------------------------------------------------------------------
MASK_ID   = 0       # embody.py
CHUNK_PAD = -1.0    # dof.py
SLOT_PAD  = -1.0    # dof.py

# ---------------------------------------------------------------------------
# Small dimensions to keep compilation fast
# ---------------------------------------------------------------------------
B       = 1       # batch size
W       = 1       # window size
N       = 4       # transformer tokens per window
E       = 64      # token embedding dim
H       = 6       # max_horizon (small for speed)
max_A   = 4       # max_dofs   (small for speed)
READOUT = "action"

FLOW_STEPS = 2    # keep Euler steps minimal for most tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def head():
    """Instantiate XFlowHead with small dims."""
    return XFlowHead(
        readout_key=READOUT,
        max_dofs=max_A,
        max_horizon=H,
        flow_steps=FLOW_STEPS,
        num_query_channels=32,
        num_heads=2,
        num_self_attend_layers=1,
        widening_factor=2,
        use_guidance=False,
    )


@pytest.fixture(scope="module")
def transformer_outputs():
    """Fake transformer outputs: {readout_key: TokenGroup(B, W, N, E)}."""
    tokens = jnp.ones((B, W, N, E), dtype=jnp.float32)
    mask   = jnp.ones((B, W, N),    dtype=jnp.int32)
    return {READOUT: TokenGroup(tokens=tokens, mask=mask)}


@pytest.fixture(scope="module")
def dof_ids():
    """(B, max_A) — first 3 valid DOFs, last one MASK-padded."""
    return jnp.array([[1, 2, 3, MASK_ID]], dtype=jnp.int32)


@pytest.fixture(scope="module")
def chunk_steps():
    """(B, max_H) — first 4 valid steps, last 2 CHUNK_PAD."""
    return jnp.array(
        [[0.0, 1.0, 2.0, 3.0, CHUNK_PAD, CHUNK_PAD]], dtype=jnp.float32
    )


@pytest.fixture(scope="module")
def slot_pos():
    """(B, max_A) — ordinal positions, last one SLOT_PAD."""
    return jnp.array([[0.0, 1.0, 2.0, SLOT_PAD]], dtype=jnp.float32)


@pytest.fixture(scope="module")
def bound_head(head, rng, transformer_outputs, dof_ids, chunk_steps, slot_pos):
    """Initialize XFlowHead params and return a bound (stateful) head."""
    init_rng, dropout_rng = jax.random.split(rng)

    dummy_time = jnp.zeros((B, W, 1),        dtype=jnp.float32)
    dummy_a_t  = jnp.zeros((B, W, H, max_A), dtype=jnp.float32)

    variables = head.init(
        {"params": init_rng, "dropout": dropout_rng},
        transformer_outputs,
        time=dummy_time,
        a_t=dummy_a_t,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        slot_pos=slot_pos,
        train=False,
    )

    return head.bind(variables)


@pytest.fixture(scope="module")
def obs(rng, transformer_outputs, dof_ids, chunk_steps, slot_pos):
    """obs dict as expected by guided_inference."""
    return {
        "transformer_outputs": transformer_outputs,
        "dof_ids":             dof_ids,
        "chunk_steps":         chunk_steps,
        "rng":                 jax.random.PRNGKey(7),
        "B":                   B,
        "W":                   W,
        "slot_pos":            slot_pos,
        "guide_input":         None,
        "guidance_mask":       None,
        "train":               False,
    }


# ---------------------------------------------------------------------------
# Integration tests -- guided_inference with real XFlowHead
# ---------------------------------------------------------------------------

class TestGuidedInferenceReal:

    def test_output_shape(self, bound_head, obs):
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)
        out = guided_inference(
            bound_head, obs, A_prev, d=1, s=3, flow_steps=FLOW_STEPS
        )
        assert out.shape == (B, W, H, max_A), f"got {out.shape}"

    def test_output_dtype(self, bound_head, obs):
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)
        out = guided_inference(
            bound_head, obs, A_prev, d=1, s=3, flow_steps=FLOW_STEPS
        )
        assert out.dtype == jnp.float32

    def test_output_finite(self, bound_head, obs):
        """No NaN or Inf in the output."""
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)
        out = guided_inference(
            bound_head, obs, A_prev, d=1, s=3, flow_steps=FLOW_STEPS
        )
        assert jnp.all(jnp.isfinite(out)), "output contains NaN or Inf"

    def test_nonzero_A_prev_affects_output(self, bound_head, obs):
        """Guidance from A_prev should influence the output."""
        A_zero = jnp.zeros((H, max_A), dtype=jnp.float32)
        A_ones = jnp.ones((H, max_A),  dtype=jnp.float32)

        out_zero = guided_inference(
            bound_head, obs, A_zero, d=1, s=3, flow_steps=FLOW_STEPS
        )
        out_ones = guided_inference(
            bound_head, obs, A_ones, d=1, s=3, flow_steps=FLOW_STEPS
        )
        assert not jnp.allclose(out_zero, out_ones), \
            "A_prev has no effect on output — guidance may be broken"

    def test_short_A_prev(self, bound_head, obs):
        """A_prev shorter than H should be padded without error."""
        A_prev = jnp.zeros((H - 2, max_A), dtype=jnp.float32)
        out = guided_inference(
            bound_head, obs, A_prev, d=1, s=3, flow_steps=FLOW_STEPS
        )
        assert out.shape == (B, W, H, max_A)

    def test_different_rng_gives_different_output(self, bound_head, obs):
        """Stochastic init (A^0 ~ N(0,I)) should vary across different RNGs."""
        obs1   = {**obs, "rng": jax.random.PRNGKey(1)}
        obs2   = {**obs, "rng": jax.random.PRNGKey(2)}
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)

        out1 = guided_inference(bound_head, obs1, A_prev, d=1, s=3, flow_steps=FLOW_STEPS)
        out2 = guided_inference(bound_head, obs2, A_prev, d=1, s=3, flow_steps=FLOW_STEPS)
        assert not jnp.allclose(out1, out2), \
            "different RNGs produced identical outputs"


# ---------------------------------------------------------------------------
# Integration tests -- full RTC lifecycle with real XFlowHead
# ---------------------------------------------------------------------------

class TestRTCReal:

    def _make_rtc(self, bound_head):
        return RTC(
            pi=bound_head,
            H=H,
            max_A=max_A,
            s_min=2,
            b=5,
            d_init=1,
            flow_steps=FLOW_STEPS,
            beta=5.0,
        )

    def test_start_stop(self, bound_head):
        rtc = self._make_rtc(bound_head)
        rtc.start()
        time.sleep(0.1)
        rtc.stop()
        assert not rtc._thread.is_alive()

    def test_get_action_shape(self, bound_head, obs):
        rtc    = self._make_rtc(bound_head)
        action = rtc.get_action(obs)
        assert action.shape == (max_A,), f"got {action.shape}"

    def test_get_action_finite(self, bound_head, obs):
        rtc    = self._make_rtc(bound_head)
        action = rtc.get_action(obs)
        assert np.all(np.isfinite(action)), "action contains NaN or Inf"

    def test_inference_loop_updates_A_cur(self, bound_head, obs):
        """After inference thread fires, _A_cur must differ from zeros."""
        rtc      = self._make_rtc(bound_head)
        original = rtc._A_cur.copy()

        rtc.start()
        for _ in range(4):
            rtc.get_action(obs)
            time.sleep(0.05)

        time.sleep(3.0)
        rtc.stop()

        assert not np.array_equal(rtc._A_cur, original), \
            "_A_cur unchanged after inference — thread may not have fired"

    def test_multiple_get_action_calls(self, bound_head, obs):
        """H+2 sequential calls must not raise IndexError."""
        rtc     = self._make_rtc(bound_head)
        actions = [rtc.get_action(obs) for _ in range(H + 2)]

        assert len(actions) == H + 2
        for a in actions:
            assert a.shape == (max_A,)
            assert np.all(np.isfinite(a))

    def test_running_get_action_no_crash(self, bound_head, obs):
        """get_action must be safe to call while inference thread is running."""
        rtc    = self._make_rtc(bound_head)
        rtc.start()
        errors = []

        for _ in range(6):
            try:
                rtc.get_action(obs)
            except Exception as e:
                errors.append(e)
            time.sleep(0.02)

        rtc.stop()
        assert errors == [], f"errors during live get_action: {errors}"


# ---------------------------------------------------------------------------
# TestChunkContinuity — paper Section 3.1, Figure 3
# ---------------------------------------------------------------------------

# Paper Table 4 hyperparameters (larger flow_steps for continuity tests)
_CC_FLOW_STEPS = 5
_CC_BETA       = 5.0
_CC_D          = 1
_CC_S          = 3
_CC_SEEDS      = [0, 1, 2, 3, 4]


def _cc_region_l2(A_new: np.ndarray, A_prev: np.ndarray, lo: int, hi: int) -> float:
    """Mean L2 distance between A_new[lo:hi] and A_prev[lo:hi]."""
    diff = A_new[lo:hi] - A_prev[lo:hi]
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


class TestChunkContinuity:
    """Paper Section 3.1, Figure 3 — chunk continuity ordering test.

    Verifies the core RTC claim: the soft mask guidance correctly steers
    A_new so that region-wise L2 distances to A_prev satisfy

        frozen_error < intermediate_error < fresh_error

    This tests the mechanism (guidance math + mask weights), not policy quality.
    Random XFlowHead weights are sufficient because the guidance term is a
    mathematical property of the inpainting algorithm.

    Parameters follow paper Table 4:
        flow_steps = 5, beta = 5.0, d = 1, s = 3
    """

    @pytest.fixture(scope="class")
    def A_prev(self):
        """(H, max_A) constant signal at 2.0 — deterministic, non-zero."""
        return jnp.ones((H, max_A), dtype=jnp.float32) * 2.0

    def _run_one_seed(self, bound_head, obs, A_prev, seed: int) -> np.ndarray:
        obs_seeded = {**obs, "rng": jax.random.PRNGKey(seed)}
        out = guided_inference(
            bound_head,
            obs_seeded,
            A_prev,
            d=_CC_D,
            s=_CC_S,
            flow_steps=_CC_FLOW_STEPS,
            beta=_CC_BETA,
        )
        return np.array(out[0, 0])

    def test_ordering_all_seeds(self, bound_head, obs, A_prev):
        """frozen_error < intermediate_error < fresh_error for every seed.

        Region boundaries (D=1, S=3, H=6):
            frozen:       A_new[0:1]  (weight = 1)
            intermediate: A_new[1:3]  (weight decays 1→0)
            fresh:        A_new[3:6]  (weight = 0, no guidance)
        """
        A_prev_np = np.array(A_prev)
        failures  = []

        for seed in _CC_SEEDS:
            A_new              = self._run_one_seed(bound_head, obs, A_prev, seed)
            frozen_error       = _cc_region_l2(A_new, A_prev_np, lo=0,        hi=_CC_D)
            intermediate_error = _cc_region_l2(A_new, A_prev_np, lo=_CC_D,    hi=H-_CC_S)
            fresh_error        = _cc_region_l2(A_new, A_prev_np, lo=H-_CC_S,  hi=H)

            if not (frozen_error < intermediate_error < fresh_error):
                failures.append(
                    f"seed={seed}: frozen={frozen_error:.4f} "
                    f"intermediate={intermediate_error:.4f} "
                    f"fresh={fresh_error:.4f} — ordering VIOLATED"
                )

        assert not failures, (
            f"Chunk continuity ordering violated in "
            f"{len(failures)}/{len(_CC_SEEDS)} seeds:\n" + "\n".join(failures)
        )

    def test_frozen_region_closest_to_A_prev(self, bound_head, obs, A_prev):
        """Frozen region must be the closest to A_prev across all seeds."""
        A_prev_np = np.array(A_prev)

        for seed in _CC_SEEDS:
            A_new        = self._run_one_seed(bound_head, obs, A_prev, seed)
            frozen_error = _cc_region_l2(A_new, A_prev_np, lo=0,       hi=_CC_D)
            fresh_error  = _cc_region_l2(A_new, A_prev_np, lo=H-_CC_S, hi=H)

            assert frozen_error < fresh_error, (
                f"seed={seed}: frozen_error ({frozen_error:.4f}) >= "
                f"fresh_error ({fresh_error:.4f}) — "
                f"guidance is not pulling frozen region toward A_prev"
            )

    def test_fresh_region_farthest_from_A_prev(self, bound_head, obs, A_prev):
        """Fresh region (weight=0) must be farthest from A_prev across all seeds."""
        A_prev_np = np.array(A_prev)

        for seed in _CC_SEEDS:
            A_new              = self._run_one_seed(bound_head, obs, A_prev, seed)
            frozen_error       = _cc_region_l2(A_new, A_prev_np, lo=0,       hi=_CC_D)
            intermediate_error = _cc_region_l2(A_new, A_prev_np, lo=_CC_D,   hi=H-_CC_S)
            fresh_error        = _cc_region_l2(A_new, A_prev_np, lo=H-_CC_S, hi=H)

            assert fresh_error > frozen_error and fresh_error > intermediate_error, (
                f"seed={seed}: fresh_error ({fresh_error:.4f}) is not the maximum — "
                f"frozen={frozen_error:.4f}, intermediate={intermediate_error:.4f}"
            )

    def test_errors_are_finite(self, bound_head, obs, A_prev):
        """All region errors must be finite (no NaN/Inf from guidance)."""
        A_prev_np = np.array(A_prev)

        for seed in _CC_SEEDS:
            A_new = self._run_one_seed(bound_head, obs, A_prev, seed)
            assert np.all(np.isfinite(A_new)), (
                f"seed={seed}: A_new contains NaN or Inf"
            )
            for lo, hi in [(0, _CC_D), (_CC_D, H-_CC_S), (H-_CC_S, H)]:
                err = _cc_region_l2(A_new, A_prev_np, lo=lo, hi=hi)
                assert np.isfinite(err), (
                    f"seed={seed}: region [{lo}:{hi}] error is not finite: {err}"
                )

    def test_region_errors_positive(self, bound_head, obs, A_prev):
        """All region errors must be strictly positive — A_new != A_prev anywhere.

        If guidance collapses A_new to A_prev even in the fresh region,
        something is wrong (e.g., velocity field is identically zero).
        """
        A_prev_np = np.array(A_prev)

        for seed in _CC_SEEDS:
            A_new       = self._run_one_seed(bound_head, obs, A_prev, seed)
            fresh_error = _cc_region_l2(A_new, A_prev_np, lo=H-_CC_S, hi=H)
            assert fresh_error > 1e-6, (
                f"seed={seed}: fresh region error ~0 ({fresh_error:.2e}) — "
                f"velocity field may be degenerate"
            )


# ---------------------------------------------------------------------------
# TestLatency — wall-clock measurements for guided_inference
# ---------------------------------------------------------------------------

class TestLatency:
    """Latency measurements for guided_inference vs. vanilla forward pass.

    No checkpoint needed — latency depends on architecture and hardware,
    not on learned weights.

    The only structural assert is that RTC is slower than a vanilla forward
    pass due to VJP overhead at each denoising step, matching the ~2-3x
    overhead reported in the paper (Table 3). All timing numbers are printed
    via -s for informational purposes.
    """

    WARMUP_CALLS  = 3
    MEASURE_CALLS = 10

    def _time_fn(self, fn, n_warmup: int, n_measure: int):
        """Run fn with warmup, return (mean_ms, std_ms)."""
        for _ in range(n_warmup):
            fn()
        jax.effects_barrier()

        times = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            fn()
            jax.effects_barrier()   # flush JAX async dispatch before measuring
            times.append((time.perf_counter() - t0) * 1000)

        return float(np.mean(times)), float(np.std(times))

    def test_guided_inference_latency(self, bound_head, obs):
        """Measure and report guided_inference wall-clock time."""
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)

        def fn():
            return guided_inference(
                bound_head, obs, A_prev, d=1, s=3, flow_steps=FLOW_STEPS
            )

        mean_ms, std_ms = self._time_fn(fn, self.WARMUP_CALLS, self.MEASURE_CALLS)
        print(
            f"\n[Latency] guided_inference: {mean_ms:.1f} ± {std_ms:.1f} ms"
            f" (flow_steps={FLOW_STEPS})"
        )
        # Sanity bound only — must finish within 60 seconds
        assert mean_ms < 60_000, f"guided_inference too slow: {mean_ms:.1f} ms"

    def test_vanilla_vs_rtc_latency(self, bound_head, obs):
        """RTC must be slower than a vanilla forward pass due to VJP overhead.

        XFlowHead.apply runs one forward pass with no gradients.
        guided_inference calls jax.vjp at every denoising step, adding
        significant overhead — expected 2-3x slowdown per denoising step
        (paper Table 3).
        """
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)

        module, variables = bound_head.unbind()
        dummy_time = jnp.zeros((B, W, 1),        dtype=jnp.float32)
        dummy_a_t  = jnp.zeros((B, W, H, max_A), dtype=jnp.float32)

        dof_ids_val     = obs["dof_ids"]
        chunk_steps_val = obs["chunk_steps"]
        slot_pos_val    = obs.get("slot_pos")

        def vanilla_fn():
            return module.apply(
                variables,
                obs["transformer_outputs"],
                dummy_time,
                dummy_a_t,
                dof_ids=dof_ids_val,
                chunk_steps=chunk_steps_val,
                slot_pos=slot_pos_val,
                train=False,
            )

        def rtc_fn():
            return guided_inference(
                bound_head, obs, A_prev, d=1, s=3, flow_steps=FLOW_STEPS
            )

        vanilla_mean, vanilla_std = self._time_fn(
            vanilla_fn, self.WARMUP_CALLS, self.MEASURE_CALLS
        )
        rtc_mean, rtc_std = self._time_fn(
            rtc_fn, self.WARMUP_CALLS, self.MEASURE_CALLS
        )

        print(f"\n[Latency] vanilla forward: {vanilla_mean:.1f} ± {vanilla_std:.1f} ms")
        print(f"[Latency] RTC guided:      {rtc_mean:.1f} ± {rtc_std:.1f} ms")
        print(f"[Latency] RTC overhead:    {rtc_mean / max(vanilla_mean, 1e-6):.2f}x")

        assert rtc_mean > vanilla_mean, (
            f"RTC ({rtc_mean:.1f} ms) is faster than vanilla "
            f"({vanilla_mean:.1f} ms) — VJP overhead missing?"
        )

    def test_latency_scales_with_flow_steps(self, bound_head, obs):
        """Wall-clock time must increase monotonically with flow_steps."""
        A_prev  = jnp.zeros((H, max_A), dtype=jnp.float32)
        results = {}

        for steps in [1, 2, 4]:
            def fn(s=steps):
                return guided_inference(
                    bound_head, obs, A_prev, d=1, s=3, flow_steps=s
                )
            mean_ms, _ = self._time_fn(fn, 2, 5)
            results[steps] = mean_ms
            print(f"\n[Latency] flow_steps={steps}: {mean_ms:.1f} ms")

        assert results[1] < results[2] < results[4], (
            f"Latency does not increase monotonically with flow_steps: {results}"
        )


# ---------------------------------------------------------------------------
# TestAsyncDelay — Q buffer and t-reset correctness
# ---------------------------------------------------------------------------

class TestAsyncDelay:
    """Verifies that the inference thread correctly measures and records delay.

    Algorithm 1 line 22: after t = t - s, the new t is enqueued onto Q.
    This t represents how many controller steps elapsed during inference
    (i.e., the actual observed inference delay d).

    Test strategy:
      - Start RTC, call get_action several times to trigger the inference thread.
      - Wait for inference to complete, then inspect Q for recorded delays.
      - Verify delays are non-negative, bounded by H, and t resets correctly.
    """

    def _make_rtc(self, bound_head):
        return RTC(
            pi=bound_head,
            H=H,
            max_A=max_A,
            s_min=2,
            b=5,
            d_init=1,
            flow_steps=FLOW_STEPS,
            beta=5.0,
        )

    def test_q_buffer_updated_after_inference(self, bound_head, obs):
        """After inference completes, Q must contain at least one newly recorded delay."""
        rtc       = self._make_rtc(bound_head)
        initial_q = list(rtc._Q)

        rtc.start()
        for _ in range(4):
            rtc.get_action(obs)
            time.sleep(0.05)

        time.sleep(4.0)   # wait for JAX compile + inference
        rtc.stop()

        final_q = list(rtc._Q)
        assert len(final_q) >= len(initial_q), "Q buffer shrank — something is wrong"
        assert final_q != initial_q or len(final_q) > 1, (
            f"Q buffer was never updated: {initial_q} -> {final_q}"
        )
        print(f"\n[AsyncDelay] Q buffer: {initial_q} -> {final_q}")

    def test_recorded_delay_is_non_negative(self, bound_head, obs):
        """All delays in Q must be >= 0.

        Algorithm 1 line 21: t = t - s.
        t is the controller step count when inference finishes;
        s is the count when inference started.
        t - s >= 0 always holds since inference takes non-negative time.
        """
        rtc = self._make_rtc(bound_head)
        rtc.start()

        for _ in range(6):
            rtc.get_action(obs)
            time.sleep(0.05)

        time.sleep(4.0)
        rtc.stop()

        for delay in rtc._Q:
            assert delay >= 0, f"Negative delay recorded in Q: {delay}"

        print(f"\n[AsyncDelay] Recorded delays: {list(rtc._Q)}")

    def test_delay_bounded_by_h(self, bound_head, obs):
        """All delays in Q must be strictly less than H.

        A delay >= H means the entire chunk has already been consumed
        before the new one was ready — RTC has fallen too far behind.
        """
        rtc = self._make_rtc(bound_head)
        rtc.start()

        for _ in range(6):
            rtc.get_action(obs)
            time.sleep(0.05)

        time.sleep(4.0)
        rtc.stop()

        for delay in rtc._Q:
            assert delay < H, (
                f"Delay ({delay}) >= H ({H}) — chunk fully consumed before "
                f"new one was ready; RTC is falling behind"
            )

    def test_t_resets_correctly_after_inference(self, bound_head, obs):
        """After inference, t must satisfy 0 <= t < H (Algorithm 1 line 21).

        If t >= H, get_action would index out of bounds on _A_cur.
        """
        rtc = self._make_rtc(bound_head)
        rtc.start()

        for _ in range(6):
            rtc.get_action(obs)
            time.sleep(0.05)

        time.sleep(4.0)
        rtc.stop()

        assert 0 <= rtc._t < H, (
            f"t={rtc._t} out of valid range [0, H={H}) after inference reset"
        )
        print(f"\n[AsyncDelay] Final t after inference reset: {rtc._t}")
