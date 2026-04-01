#
# trajectory level transforms
#
from __future__ import annotations

import jax
import jax.numpy as jnp


def binarize_gripper_actions(actions: jnp.ndarray, open=0.95, close=0.05) -> jnp.ndarray:
    """
    Converts continuous gripper actions into binary (0 or 1) by scanning
    from the end of the array to the beginning, just like the original
    TensorFlow version.

    Args:
        actions: A 1D array of continuous gripper actions.
    """
    # Define masks
    open_mask = actions > open
    closed_mask = actions < close
    in_between_mask = jnp.logical_not(jnp.logical_or(open_mask, closed_mask))
    is_open_float = open_mask.astype(jnp.float32)

    # We'll scan from the last index down to the first.
    # However, JAX's scan goes forward, so we iterate over reversed indices
    # and then flip the result back.

    def reversed_scan_fn(carry, i):
        """
        carry: The "current" action value carried from a later (higher) index.
        i:     The iteration index [0..n-1], but we map it to idx = n-1 - i.
        """
        idx = actions.shape[0] - 1 - i
        # If we're in-between, keep the carry;
        # otherwise, set the carry to open (1.0) or closed (0.0) depending on is_open_float.
        new_carry = jnp.where(in_between_mask[idx], carry, is_open_float[idx])
        return new_carry, new_carry  # (next_carry, scan_output)

    n = actions.shape[0]
    init_carry = actions[-1]  # Start carry as the last element of 'actions'

    # Scan over a range of n steps; i goes from 0..n-1
    # reversed_scan_fn updates the carry and returns it as output each step
    _, reversed_carries = jax.lax.scan(reversed_scan_fn, init_carry, jnp.arange(n), reverse=True)

    # reversed_carries[0] corresponds to idx = n-1,
    # reversed_carries[1] corresponds to idx = n-2, ...
    # So flip to get new_actions in the normal 0..n-1 order
    new_actions = jnp.flip(reversed_carries)

    return new_actions


def is_noop(self, action, prev_action=None, threshold=1e-3):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """

    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return jnp.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return jnp.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def scan_noop(positions: jnp.ndarray, threshold: float = 1e-3, binary=True) -> jnp.ndarray:
    """
    Given a trajectory (positions: [n, d]), returns a boolean array of length n
    indicating whether each step is a no-op.
    """
    first = jnp.linalg.norm(positions[0, :-1]) < threshold

    def f(prev, this):
        if binary:
            noop = jnp.logical_and(
                jnp.linalg.norm(this[:-1] - prev[:-1]) < threshold,
                this[-1] == prev[-1],
            )
        else:
            noop = jnp.linalg.norm(this - prev) < threshold

        act = jnp.where(noop, prev, this)
        return act, noop

    _carry, noops = jax.lax.scan(f, positions[0], positions[1:])
    return jnp.concatenate([jnp.array([first]), noops])
