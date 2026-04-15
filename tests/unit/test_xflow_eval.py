"""Unit tests for crossformer.run.xflow_eval adapters.

Replaces the stale tests/unit/test_train_xflow.py that imported symbols
which were refactored out of scripts/train/xflow.py.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from crossformer.embody import DOF
from crossformer.run.xflow_eval import (
    adapt_rast_batch,
    extract_bundled_actions,
    JOINT_IDS,
    RAST_IDS,
)


def test_rast_ids_are_joints_plus_gripper():
    assert RAST_IDS[:-1] == JOINT_IDS
    assert RAST_IDS[-1] == DOF["gripper"]
    assert len(RAST_IDS) == 8


def test_extract_bundled_actions_shapes():
    b, h, a = 2, 4, 5
    batch = {
        "act": {
            "base": jnp.zeros((b, h, a), dtype=jnp.float32),
            "id": jnp.zeros((b, a), dtype=jnp.int32),
        }
    }
    actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h=h)
    assert actions.shape == (b, 1, h, a)
    assert dof_ids.shape == (b, a)
    assert chunk_steps.shape == (b, h)
    assert jnp.array_equal(chunk_steps[0], jnp.arange(h, dtype=jnp.float32))


def test_adapt_rast_batch_maps_slots_to_canonical():
    b, h, a = 2, 3, 4
    flow_steps = 2
    # batch 0: slot 0 -> j0, slot 1 -> gripper. batch 1: only slot 2 -> j3.
    dof_ids = np.array(
        [[DOF["j0"], DOF["gripper"], 0, 0], [0, 0, DOF["j3"], 0]],
        dtype=np.int32,
    )
    base = np.arange(b * h * a, dtype=np.float32).reshape(b, h, a)
    flow = np.arange(flow_steps * b * 1 * h * a, dtype=np.float32).reshape(flow_steps, b, 1, h, a)
    act = {"base": jnp.asarray(base), "id": jnp.asarray(dof_ids)}
    out, keep = adapt_rast_batch(act, jnp.asarray(flow))
    assert out is not None
    # both rows have at least one target DOF
    assert list(keep) == [0, 1]
    canonical = out["act"]["base"]
    # canonical dim is len(RAST_IDS) = 8
    assert canonical.shape[-1] == len(RAST_IDS)
    # batch 0 slot 0 -> canonical j0 (index 0 in RAST_IDS)
    assert np.allclose(canonical[0, :, :, 0], base[0, :, 0][:, None].squeeze(-1), equal_nan=True) or np.allclose(
        canonical[0, ..., 0], base[0, ..., 0]
    )


def test_adapt_rast_batch_drops_rows_with_no_targets():
    b, h, a = 2, 2, 3
    dof_ids = np.zeros((b, a), dtype=np.int32)  # all MASK
    base = np.zeros((b, h, a), dtype=np.float32)
    flow = np.zeros((1, b, 1, h, a), dtype=np.float32)
    act = {"base": jnp.asarray(base), "id": jnp.asarray(dof_ids)}
    out, keep = adapt_rast_batch(act, jnp.asarray(flow))
    assert out is None and keep is None
