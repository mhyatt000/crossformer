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

FLOW_STEPS = 2    # keep Euler steps minimal


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
        num_query_channels=32,   # small for speed
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
    ids = jnp.array([[1, 2, 3, MASK_ID]], dtype=jnp.int32)  # (1, 4)
    return ids


@pytest.fixture(scope="module")
def chunk_steps():
    """(B, max_H) — first 4 valid steps, last 2 CHUNK_PAD."""
    steps = jnp.array(
        [[0.0, 1.0, 2.0, 3.0, CHUNK_PAD, CHUNK_PAD]], dtype=jnp.float32
    )
    return steps


@pytest.fixture(scope="module")
def slot_pos():
    """(B, max_A) — ordinal positions, last one SLOT_PAD."""
    pos = jnp.array([[0.0, 1.0, 2.0, SLOT_PAD]], dtype=jnp.float32)
    return pos


@pytest.fixture(scope="module")
def bound_head(head, rng, transformer_outputs, dof_ids, chunk_steps, slot_pos):
    """Initialize XFlowHead params and return a bound (stateful) head."""
    init_rng, dropout_rng = jax.random.split(rng)

    dummy_time = jnp.zeros((B, W, 1), dtype=jnp.float32)
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
        A_ones = jnp.ones((H, max_A), dtype=jnp.float32)

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
        obs1 = {**obs, "rng": jax.random.PRNGKey(1)}
        obs2 = {**obs, "rng": jax.random.PRNGKey(2)}
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
        rtc = self._make_rtc(bound_head)
        action = rtc.get_action(obs)
        assert action.shape == (max_A,), f"got {action.shape}"

    def test_get_action_finite(self, bound_head, obs):
        rtc = self._make_rtc(bound_head)
        action = rtc.get_action(obs)
        assert np.all(np.isfinite(action)), "action contains NaN or Inf"

    def test_inference_loop_updates_A_cur(self, bound_head, obs):
        """After inference thread fires, _A_cur must differ from zeros."""
        rtc = self._make_rtc(bound_head)
        original = rtc._A_cur.copy()

        rtc.start()
        # Trigger s_min calls so the inference loop wakes up
        for _ in range(4):
            rtc.get_action(obs)
            time.sleep(0.05)

        # Wait for inference to complete (real JAX compile + run)
        time.sleep(3.0)
        rtc.stop()

        assert not np.array_equal(rtc._A_cur, original), \
            "_A_cur unchanged after inference — thread may not have fired"

    def test_multiple_get_action_calls(self, bound_head, obs):
        """H+2 sequential calls must not raise IndexError."""
        rtc = self._make_rtc(bound_head)
        actions = []
        for _ in range(H + 2):
            actions.append(rtc.get_action(obs))

        assert len(actions) == H + 2
        for a in actions:
            assert a.shape == (max_A,)
            assert np.all(np.isfinite(a))

    def test_running_get_action_no_crash(self, bound_head, obs):
        """get_action must be safe to call while inference thread is running."""
        rtc = self._make_rtc(bound_head)
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
