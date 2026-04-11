"""Shared pytest fixtures for RTC tests.

Provides:
  - bound_head, obs: random-weight XFlowHead for unit/integration tests.
  - checkpoint_pi, dataset_obs_list: real checkpoint + dataset fixtures
    for eval tests. Activated via --checkpoint and --dataset CLI args.
    Tests using these fixtures are skipped if args are not provided.

Run all tests (random weights only):
    pytest crossformer/model/components/rtc/ -v

Run with real checkpoint + dataset:
    pytest crossformer/model/components/rtc/ -v \
        --checkpoint ~/projects/crossformer/0403_super-night-806/params \
        --dataset ~/.cache/arrayrecords/xgym_sweep_single/0.5.2/to_step \
        --n_steps 20
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.xflow import XFlowHead

# ---------------------------------------------------------------------------
# Constants matching dof.py / embody.py
# ---------------------------------------------------------------------------
MASK_ID   = 0
CHUNK_PAD = -1.0
SLOT_PAD  = -1.0

# ---------------------------------------------------------------------------
# Small dims for random-weight tests
# ---------------------------------------------------------------------------
B     = 1
W     = 1
N     = 4
E     = 64
H     = 6
max_A = 4

READOUT    = "action"
FLOW_STEPS = 2


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--checkpoint", default=None,
                     help="Path to CrossFormer checkpoint dir.")
    parser.addoption("--dataset", default=None,
                     help="Path to ArrayRecord dataset dir.")
    parser.addoption("--n_steps", type=int, default=10,
                     help="Number of dataset steps to load (default: 10).")


# ---------------------------------------------------------------------------
# Random-weight fixtures (session scope — shared across all test files)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def head():
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
    tokens = jnp.ones((B, W, N, E), dtype=jnp.float32)
    mask   = jnp.ones((B, W, N),    dtype=jnp.int32)
    return {READOUT: TokenGroup(tokens=tokens, mask=mask)}


@pytest.fixture(scope="session")
def dof_ids():
    return jnp.array([[1, 2, 3, MASK_ID]], dtype=jnp.int32)


@pytest.fixture(scope="session")
def chunk_steps():
    return jnp.array(
        [[0.0, 1.0, 2.0, 3.0, CHUNK_PAD, CHUNK_PAD]], dtype=jnp.float32
    )


@pytest.fixture(scope="session")
def slot_pos():
    return jnp.array([[0.0, 1.0, 2.0, SLOT_PAD]], dtype=jnp.float32)


@pytest.fixture(scope="session")
def bound_head(head, rng, transformer_outputs, dof_ids, chunk_steps, slot_pos):
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


@pytest.fixture(scope="session")
def obs(rng, transformer_outputs, dof_ids, chunk_steps, slot_pos):
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
# Real checkpoint + dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def checkpoint_pi(request):
    """Bound XFlowHead from a real checkpoint.

    Skip if --checkpoint not provided.
    """
    ckpt = request.config.getoption("--checkpoint")
    if not ckpt:
        pytest.skip("--checkpoint not provided")

    from crossformer.model.crossformer_model import CrossFormerModel
    model = CrossFormerModel.load_pretrained(
        str(Path(ckpt).expanduser()), step=None
    )
    return model.module.bind({"params": model.params}).heads["xflow"], model


@pytest.fixture(scope="session")
def dataset_obs_list(request, checkpoint_pi):
    """List of (obs_dict, action_np) from real dataset.

    Skip if --dataset not provided.
    Each obs_dict has transformer_outputs ready for guided_inference.
    """
    dataset = request.config.getoption("--dataset")
    if not dataset:
        pytest.skip("--dataset not provided")

    n_steps = request.config.getoption("--n_steps")
    pi, model = checkpoint_pi

    from crossformer.model.components.rtc.eval_rtc_checkpoint_offline import (
        load_obs_list,
        _make_obs,
    )

    raw_list = load_obs_list(model, dataset, n_steps)

    # Convert to (obs_dict_with_transformer_outputs, action_np)
    result = []
    for obs_raw, task_raw, action in raw_list:
        obs_dict = _make_obs(model, obs_raw, task_raw, jax.random.PRNGKey(0))
        result.append((obs_dict, action))
    return result
