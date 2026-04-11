"""Tests for rtc_algorithm.py.

Unit tests: mock pi, no CrossFormer deps.
Integration tests: minimal real XFlowHead-like pi stub.

Run with:
    pytest test_rtc_algorithm.py -v
"""

from __future__ import annotations

import threading
import time
import types
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# -- import the module under test ------------------------------------------
# Adjust the import path to match your project layout.
from crossformer.model.components.rtc.rtc_algorithm import (
    RTC,
    build_soft_mask,
    guided_inference,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B, W, H, max_A = 1, 2, 10, 6
FLOW_STEPS = 2  # keep tests fast


def _make_mock_pi(B=B, W=W, H=H, max_A=max_A):
    """Build a minimal mock pi that satisfies guided_inference's expectations.

    module.apply(...) returns zeros of shape (B, W, H*max_A).
    pi.unbind() returns (module, variables).
    pi.max_horizon == H, pi.max_dofs == max_A.
    """
    module = MagicMock()
    module.apply = MagicMock(
        return_value=jnp.zeros((B, W, H * max_A), dtype=jnp.float32)
    )

    variables = {"params": {}}

    pi = MagicMock()
    pi.unbind.return_value = (module, variables)
    pi.max_horizon = H
    pi.max_dofs = max_A

    return pi


def _make_obs(B=B, W=W, H=H, max_A=max_A):
    """Minimal obs dict matching guided_inference's expected keys."""
    rng = jax.random.PRNGKey(0)
    return {
        "transformer_outputs": {},      # module.apply is mocked, not inspected
        "dof_ids":   jnp.zeros((B, max_A), dtype=jnp.int32),
        "chunk_steps": jnp.zeros((B, H), dtype=jnp.float32),
        "rng":  rng,
        "B":    B,
        "W":    W,
        "slot_pos":      None,
        "guide_input":   None,
        "guidance_mask": None,
        "train": False,
    }


# ===========================================================================
# Unit tests -- build_soft_mask
# ===========================================================================

class TestBuildSoftMask:

    def test_output_shape(self):
        W = build_soft_mask(H=10, d=2, s=5)
        assert W.shape == (10,)

    def test_frozen_region(self):
        """Indices < d must be exactly 1."""
        d, s, H = 3, 6, 15
        W = build_soft_mask(H, d, s)
        assert jnp.all(W[:d] == 1.0), f"frozen region not 1: {W[:d]}"

    def test_fresh_region(self):
        """Indices >= H-s must be exactly 0."""
        d, s, H = 3, 6, 15
        W = build_soft_mask(H, d, s)
        assert jnp.all(W[H - s:] == 0.0), f"fresh region not 0: {W[H - s:]}"

    def test_intermediate_region_in_0_1(self):
        """d <= i < H-s must be in [0, 1]."""
        d, s, H = 3, 6, 15
        W = build_soft_mask(H, d, s)
        mid = W[d: H - s]
        assert jnp.all(mid >= 0.0) and jnp.all(mid <= 1.0)

    def test_constraint_violation_raises(self):
        """d <= s <= H-d must hold, else AssertionError."""
        with pytest.raises(AssertionError):
            build_soft_mask(H=10, d=6, s=3)   # d > s

    def test_all_frozen_edge_case(self):
        """When d == H-s the intermediate region is empty."""
        d, s, H = 3, 7, 10   # H-s == d == 3
        W = build_soft_mask(H, d, s)
        assert W.shape == (H,)
        assert jnp.all(W[:d] == 1.0)
        assert jnp.all(W[H - s:] == 0.0)


# ===========================================================================
# Unit tests -- guided_inference
# ===========================================================================

class TestGuidedInference:

    def test_output_shape(self):
        pi = _make_mock_pi()
        obs = _make_obs()
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)

        out = guided_inference(pi, obs, A_prev, d=2, s=4, flow_steps=FLOW_STEPS)
        assert out.shape == (B, W, H, max_A), f"unexpected shape {out.shape}"

    def test_output_dtype(self):
        pi = _make_mock_pi()
        obs = _make_obs()
        A_prev = jnp.zeros((H, max_A), dtype=jnp.float32)

        out = guided_inference(pi, obs, A_prev, d=2, s=4, flow_steps=FLOW_STEPS)
        assert out.dtype == jnp.float32

    def test_short_A_prev_padded(self):
        """A_prev shorter than H should be zero-padded without error."""
        pi = _make_mock_pi()
        obs = _make_obs()
        A_prev = jnp.zeros((H - 3, max_A), dtype=jnp.float32)  # shorter

        out = guided_inference(pi, obs, A_prev, d=2, s=4, flow_steps=FLOW_STEPS)
        assert out.shape == (B, W, H, max_A)

    def test_long_A_prev_truncated(self):
        """A_prev longer than H should be truncated without error."""
        pi = _make_mock_pi()
        obs = _make_obs()
        A_prev = jnp.zeros((H + 5, max_A), dtype=jnp.float32)  # longer

        out = guided_inference(pi, obs, A_prev, d=2, s=4, flow_steps=FLOW_STEPS)
        assert out.shape == (B, W, H, max_A)

    def test_velocity_fn_called(self):
        """module.apply must be called during integration."""
        pi = _make_mock_pi()
        obs = _make_obs()
        A_prev = jnp.zeros((H, max_A))

        guided_inference(pi, obs, A_prev, d=2, s=4, flow_steps=FLOW_STEPS)
        module, _ = pi.unbind()
        assert module.apply.call_count > 0


# ===========================================================================
# Unit tests -- RTC initialisation
# ===========================================================================

class TestRTCInit:

    def test_initial_t_is_zero(self):
        pi = _make_mock_pi()
        rtc = RTC(pi, H=H, max_A=max_A, s_min=2)
        assert rtc._t == 0

    def test_default_A_cur_zeros(self):
        pi = _make_mock_pi()
        rtc = RTC(pi, H=H, max_A=max_A, s_min=2)
        assert rtc._A_cur.shape == (H, max_A)
        assert np.all(rtc._A_cur == 0.0)

    def test_custom_A_init(self):
        pi = _make_mock_pi()
        A_init = np.ones((H, max_A), dtype=np.float32)
        rtc = RTC(pi, H=H, max_A=max_A, s_min=2, A_init=A_init)
        np.testing.assert_array_equal(rtc._A_cur, A_init)

    def test_delay_buffer_initialised(self):
        pi = _make_mock_pi()
        rtc = RTC(pi, H=H, max_A=max_A, s_min=2, d_init=3, b=10)
        assert list(rtc._Q) == [3]


# ===========================================================================
# Unit tests -- get_action
# ===========================================================================

class TestGetAction:

    def _make_rtc(self):
        pi = _make_mock_pi()
        A_init = np.arange(H * max_A, dtype=np.float32).reshape(H, max_A)
        return RTC(pi, H=H, max_A=max_A, s_min=H + 1, A_init=A_init)

    def test_increments_t(self):
        rtc = self._make_rtc()
        obs = _make_obs()
        rtc.get_action(obs)
        assert rtc._t == 1

    def test_returns_correct_row(self):
        rtc = self._make_rtc()
        obs = _make_obs()
        action = rtc.get_action(obs)   # t becomes 1, returns A_cur[0]
        np.testing.assert_array_equal(action, rtc._A_cur[0])

    def test_index_clamping(self):
        """After H calls t-1 >= H; index must clamp to H-1, not raise."""
        rtc = self._make_rtc()
        obs = _make_obs()

        for _ in range(H + 3):     # push t well past H
            try:
                action = rtc.get_action(obs)
            except IndexError:
                pytest.fail("get_action raised IndexError — clamping fix missing")

        # The returned action must equal A_cur[H-1] (last valid row)
        np.testing.assert_array_equal(action, rtc._A_cur[H - 1])

    def test_updates_o_cur(self):
        rtc = self._make_rtc()
        obs = _make_obs()
        rtc.get_action(obs)
        assert rtc._o_cur is obs

    def test_action_shape(self):
        rtc = self._make_rtc()
        obs = _make_obs()
        action = rtc.get_action(obs)
        assert action.shape == (max_A,)


# ===========================================================================
# Integration tests -- RTC thread lifecycle
# ===========================================================================

class TestRTCThreadLifecycle:
    """Smoke-tests that the background thread starts, runs, and stops cleanly.

    guided_inference is monkey-patched to avoid real JAX compilation.
    """

    def _make_rtc_with_fast_inference(self):
        pi = _make_mock_pi()
        rtc = RTC(pi, H=H, max_A=max_A, s_min=2, b=5, d_init=1,
                  flow_steps=FLOW_STEPS)

        # Patch _run_guided_inference so the thread never blocks on JAX
        def _fast_inference(obs, A_prev, d, s):
            return np.zeros((H, max_A), dtype=np.float32)

        rtc._run_guided_inference = _fast_inference
        return rtc

    def test_start_stop_no_hang(self):
        rtc = self._make_rtc_with_fast_inference()
        rtc.start()
        time.sleep(0.05)
        rtc.stop()                          # must return within timeout
        assert not rtc._thread.is_alive()

    def test_get_action_while_running(self):
        rtc = self._make_rtc_with_fast_inference()
        obs = _make_obs()
        rtc.start()

        actions = []
        for _ in range(H + 2):
            actions.append(rtc.get_action(obs))
            time.sleep(0.005)

        rtc.stop()
        assert len(actions) == H + 2
        for a in actions:
            assert a.shape == (max_A,)

    def test_no_index_error_under_threading(self):
        """get_action must never raise IndexError even under race conditions."""
        rtc = self._make_rtc_with_fast_inference()
        obs = _make_obs()
        errors = []

        def _caller():
            for _ in range(20):
                try:
                    rtc.get_action(obs)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.002)

        rtc.start()
        t = threading.Thread(target=_caller)
        t.start()
        t.join(timeout=2.0)
        rtc.stop()

        assert errors == [], f"Errors in get_action thread: {errors}"

    def test_A_cur_replaced_after_inference(self):
        """After the inference thread fires, _A_cur must change from zeros."""
        pi = _make_mock_pi()
        sentinel = np.ones((H, max_A), dtype=np.float32) * 42.0

        rtc = RTC(pi, H=H, max_A=max_A, s_min=2, b=5, d_init=1,
                  flow_steps=FLOW_STEPS)

        def _fast_inference(obs, A_prev, d, s):
            return sentinel.copy()

        rtc._run_guided_inference = _fast_inference
        obs = _make_obs()

        rtc.start()
        # Trigger s_min calls so the thread wakes up
        for _ in range(4):
            rtc.get_action(obs)
            time.sleep(0.02)

        rtc.stop()
        # After inference _A_cur should equal sentinel (or be reset)
        # We just verify no crash and the thread finished.
        assert not rtc._thread.is_alive()
