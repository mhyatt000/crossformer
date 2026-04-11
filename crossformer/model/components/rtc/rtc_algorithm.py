"""Real-Time Chunking (RTC) for CrossFormer XFlowHead policies.

Implements Algorithm 1 from:
  "Real-Time Execution of Action Chunking Flow Policies"
  Black, Galliker, Levine -- NeurIPS 2025

Public API
----------
build_soft_mask(H, d, s)       Equation 5 -- soft mask weights W.
guided_inference(pi, ...)      GUIDEDINFERENCE (Algorithm 1, lines 23-29).
RTC                            Full async system (GETACTION + INFERENCELOOP).
configure_timing(dt)           Set controller sampling period in seconds.
"""

from __future__ import annotations

import threading
import time as _time_module
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Equation 5 -- soft mask
# ---------------------------------------------------------------------------

def build_soft_mask(H: int, d: int, s: int) -> jax.Array:
    """Build soft mask weights W of shape (H,) per Equation 5.

    W[i] = 1            if i < d          (frozen, will execute)
    W[i] = exp decay    if d <= i < H-s   (intermediate, soft)
    W[i] = 0            if i >= H-s       (fresh, no prev chunk)
    """
    assert d <= s <= H - d, f"Constraint d<=s<=H-d violated: d={d} s={s} H={H}"

    indices = jnp.arange(H, dtype=jnp.float32)

    denom = float(H - s - d + 1)
    c = (H - s - indices) / denom
    c_clipped = jnp.clip(c, 0.0, 1.0)
    decay = c_clipped * (jnp.exp(c_clipped) - 1.0) / (jnp.e - 1.0)

    frozen = (indices < d).astype(jnp.float32)
    soft   = jnp.logical_and(indices >= d, indices < H - s).astype(jnp.float32)
    W = frozen + soft * decay

    return W  # (H,)


# ---------------------------------------------------------------------------
# Equation 4 -- r^2_tau
# ---------------------------------------------------------------------------

def _r2(tau: float) -> float:
    """r^2_tau = (1-tau)^2 / (tau^2 + (1-tau)^2)."""
    return (1.0 - tau) ** 2 / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)


# ---------------------------------------------------------------------------
# GUIDEDINFERENCE -- Algorithm 1, lines 23-29
# ---------------------------------------------------------------------------

def guided_inference(
    pi,                          # XFlowHead (bound with params via unbind())
    obs: dict,                   # o: current observation dict
    A_prev: jax.Array,           # (H_prev, max_A) remaining prev chunk actions
    d: int,                      # inference delay (controller steps)
    s: int,                      # execution horizon
    flow_steps: int = 5,         # Euler integration steps (n in paper)
    beta: float = 5.0,           # guidance weight clipping (beta in paper)
) -> jax.Array:
    """GUIDEDINFERENCE (Algorithm 1, lines 23-29).

    Args:
        pi:        XFlowHead bound with params.
        obs:       current observation dict (o in paper). Must contain:
                     transformer_outputs, dof_ids, chunk_steps, rng, B, W,
                     and optionally slot_pos, guide_input, guidance_mask, train.
        A_prev:    (H_prev, max_A) remaining actions from prev chunk.
        d:         inference delay (controller steps).
        s:         execution horizon.
        flow_steps: Euler integration steps (n in paper).
        beta:      guidance weight clipping.

    Returns:
        A_new: (B, W, H, max_A) denoised action chunk.
    """
    module, variables = pi.unbind()
    H = pi.max_horizon
    max_A = pi.max_dofs

    # unpack o (all CrossFormer-specific args live in obs)
    transformer_outputs = obs["transformer_outputs"]
    dof_ids             = obs["dof_ids"]
    chunk_steps         = obs["chunk_steps"]
    rng                 = obs["rng"]
    B                   = obs["B"]
    W                   = obs["W"]
    slot_pos            = obs.get("slot_pos")
    guide_input         = obs.get("guide_input")
    guidance_mask       = obs.get("guidance_mask")
    train               = obs.get("train", False)

    # -- Build velocity fn from pi (the flow policy) -----------------------
    def velocity_fn(a_tau: jax.Array, tau: float) -> jax.Array:
        """v_pi(A^tau, o, tau) -- calls XFlowHead with current obs."""
        Bv, Wv = a_tau.shape[:2]
        t = jnp.full((Bv, Wv, 1), tau, dtype=a_tau.dtype)
        return module.apply(
            variables,
            transformer_outputs,
            t,
            a_tau,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            slot_pos=slot_pos,
            train=train,
            guide_input=guide_input,
            guidance_mask=guidance_mask,
        )  # (B, W, H*max_A)

    # line 24: compute W using Eq. 5
    mask = build_soft_mask(H, d, s)                   # (H,)
    W_flat = jnp.tile(
        jnp.repeat(mask, max_A)[None, None, :],
        (B, W, 1),
    )  # (B, W, H*max_A)

    # line 24: right-pad A_prev to length H
    H_prev = A_prev.shape[0]
    if H_prev < H:
        pad = jnp.zeros((H - H_prev, max_A), dtype=A_prev.dtype)
        A_prev_padded = jnp.concatenate([A_prev, pad], axis=0)
    else:
        A_prev_padded = A_prev[:H]
    Y = jnp.tile(
        A_prev_padded.reshape(-1)[None, None, :],
        (B, W, 1),
    )  # (B, W, H*max_A)

    # line 24: initialize A^0 ~ N(0, I)
    A_tau = jax.random.normal(rng, (B, W, H * max_A))

    dt = 1.0 / max(flow_steps, 1)

    # lines 25-29: for tau = 0 to 1 with step size 1/n
    for step in range(flow_steps):
        tau = (step + 0.5) * dt   # midpoint, matches XFlowHead.predict_action

        # line 26: f_{A_c1}: A' -> A' + (1 - tau) * v_pi(A', o, tau)
        def f_denoise(A_prime):
            v = velocity_fn(A_prime, tau)
            return A_prime + (1.0 - tau) * v

        # lines 27-28: weighted error + vjp
        A_c1, vjp_fn = jax.vjp(f_denoise, A_tau)
        e = (Y - A_c1) * W_flat                       # (B, W, H*max_A)
        (g,) = vjp_fn(e)                              # (B, W, H*max_A)

        # line 29: integration step (Eq. 1) with guidance (Eq. 2)
        r2 = _r2(tau)
        guidance_scale = jnp.minimum(beta, (1.0 - tau) / (tau * r2 + 1e-8))
        v_pi = velocity_fn(A_tau, tau)
        A_tau = A_tau + dt * (v_pi + guidance_scale * g)

    return A_tau.reshape(B, W, H, max_A)


# ---------------------------------------------------------------------------
# RTC -- Algorithm 1: GETACTION + INFERENCELOOP
# ---------------------------------------------------------------------------

class RTC:
    """Real-Time Chunking: async inference with soft-mask inpainting.

    Implements the full Algorithm 1 system. pi (the flow policy) is
    passed explicitly, matching GUIDEDINFERENCE(pi, o, A_prev, d, s).

    Args:
        pi:          XFlowHead (bound with params).
        H:           prediction horizon.
        max_A:       action dimension (padded DOFs).
        s_min:       minimum execution horizon (s_min in paper).
        b:           delay buffer size.
        d_init:      initial delay estimate (controller steps).
        A_init:      (H, max_A) initial action chunk A_init.
        flow_steps:  denoising steps n.
        beta:        guidance weight clipping beta.
    """

    def __init__(
        self,
        pi,
        H: int,
        max_A: int,
        s_min: int,
        b: int = 10,
        d_init: int = 1,
        A_init: np.ndarray | None = None,
        flow_steps: int = 5,
        beta: float = 5.0,
    ):
        self._pi = pi              # flow policy pi
        self._H = H
        self._max_A = max_A
        self._s_min = s_min
        self._flow_steps = flow_steps
        self._beta = beta

        # mutex M + condition variable C
        self._cond = threading.Condition()

        # INITIALIZESHAREDSTATE (Algorithm 1, line 2)
        self._t: int = 0
        if A_init is None:
            A_init = np.zeros((H, max_A), dtype=np.float32)
        self._A_cur: np.ndarray = np.array(A_init, dtype=np.float32)
        self._o_cur = None

        # line 11: Q = new Queue([d_init], maxlen=b)
        self._Q: deque[int] = deque([d_init], maxlen=b)

        self._running = False
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)

    def start(self):
        """Start the background inference thread."""
        self._running = True
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop and join."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=5.0)

    def get_action(self, obs) -> np.ndarray:
        """GETACTION -- Algorithm 1, lines 3-8.

        Called every Delta_t by the controller.

        Returns:
            action: (max_A,) action for this timestep.
        """
        with self._cond:
            self._t += 1                               # line 5
            self._o_cur = obs                          # line 6
            self._cond.notify_all()                    # line 7
            idx = min(self._t - 1, self._H - 1)
            return np.array(self._A_cur[idx]) # line 8

    def _inference_loop(self):
        """INFERENCELOOP -- Algorithm 1, lines 9-22."""
        with self._cond:
            while self._running:
                # line 13: wait on C until t >= s_min
                self._cond.wait_for(
                    lambda: self._t >= self._s_min or not self._running
                )
                if not self._running:
                    break

                s   = self._t                 # line 14: s = t
                A_prev = self._A_cur[s:].copy() # line 15: A_prev = A_cur[s..H-1]
                o = self._o_cur             # line 16: o = o_cur
                d   = max(self._Q)            # line 17: d = max(Q)

                # line 18: with M released do
                self._cond.release()
                try:
                    # line 19: A_new = GUIDEDINFERENCE(pi, o, A_prev, d, s)
                    A_new = self._run_guided_inference(o, A_prev, d, s)
                finally:
                    self._cond.acquire()

                self._A_cur = np.array(A_new, dtype=np.float32)  # line 20
                self._t = self._t - s                             # line 21
                self._Q.append(self._t)                           # line 22

    def _run_guided_inference(self, o, A_prev, d, s):
        """Calls guided_inference(pi, o, A_prev, d, s) -- line 19."""
        return guided_inference(
            pi=self._pi,
            obs=o,
            A_prev=jnp.array(A_prev),
            d=d,
            s=s,
            flow_steps=self._flow_steps,
            beta=self._beta,
        )


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

_delta_t: float = 0.02  # default 50 Hz


def configure_timing(delta_t_seconds: float):
    """Set controller sampling period Delta_t in seconds (default 0.02 = 50Hz)."""
    global _delta_t
    _delta_t = delta_t_seconds
