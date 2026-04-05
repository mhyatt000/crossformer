"""Shared pytest fixtures for RTC tests.

Provides bound_head and obs at session scope so both
test_rtc_integration.py and test_rtc_chunk_continuity.py share the same
compiled XFlowHead — JAX JIT compilation happens only once.

Run with:
    pytest crossformer/model/components/rtc/ -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.xflow import XFlowHead

# ---------------------------------------------------------------------------
# Constants matching dof.py / embody.py
# ---------------------------------------------------------------------------
MASK_ID   = 0       # embody.py
CHUNK_PAD = -1.0    # dof.py
SLOT_PAD  = -1.0    # dof.py

# ---------------------------------------------------------------------------
# Dimensions — small to keep compilation fast
# ---------------------------------------------------------------------------
B     = 1    # batch size
W     = 1    # window size
N     = 4    # transformer tokens per window
E     = 64   # token embedding dim
H     = 6    # max_horizon
max_A = 4    # max_dofs

READOUT    = "action"
FLOW_STEPS = 2   # minimal Euler steps for integration tests


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# (session scope: shared across all test files in one pytest run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def transformer_outputs():
    """Fake transformer outputs: {readout_key: TokenGroup(B, W, N, E)}."""
    tokens = jnp.ones((B, W, N, E), dtype=jnp.float32)
    mask   = jnp.ones((B, W, N),    dtype=jnp.int32)
    return {READOUT: TokenGroup(tokens=tokens, mask=mask)}


@pytest.fixture(scope="session")
def dof_ids():
    """(B, max_A) — first 3 valid DOFs, last one MASK-padded."""
    return jnp.array([[1, 2, 3, MASK_ID]], dtype=jnp.int32)


@pytest.fixture(scope="session")
def chunk_steps():
    """(B, H) — first 4 valid steps, last 2 CHUNK_PAD."""
    return jnp.array(
        [[0.0, 1.0, 2.0, 3.0, CHUNK_PAD, CHUNK_PAD]], dtype=jnp.float32
    )


@pytest.fixture(scope="session")
def slot_pos():
    """(B, max_A) — ordinal positions, last one SLOT_PAD."""
    return jnp.array([[0.0, 1.0, 2.0, SLOT_PAD]], dtype=jnp.float32)


@pytest.fixture(scope="session")
def bound_head(head, rng, transformer_outputs, dof_ids, chunk_steps, slot_pos):
    """Initialize XFlowHead params with random weights and return bound head.

    This is the single compilation point — session scope means JAX JIT
    compiles once and is reused by test_rtc_integration.py,
    test_rtc_chunk_continuity.py, and any future test files.
    """
    init_rng, dropout_rng = jax.random.split(rng)

    dummy_time = jnp.zeros((B, W, 1),       dtype=jnp.float32)
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


@pytest.fixture(scope="session")
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
