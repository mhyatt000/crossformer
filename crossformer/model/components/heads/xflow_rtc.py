"""Real-Time Chunking (RTC) for XFlowHead.

Inference-time inpainting via ΠGDM guidance (Black et al., NeurIPS 2025).
No retraining required. Does not modify XFlowHead or predict_action.

Usage:
    from crossformer.model.components.heads.xflow_rtc import rtc_predict_action

    pred = rtc_predict_action(
        bound=bound,                  # model.module.bind({"params": params})
        transformer_outputs=outputs,
        rng=rng,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        a_prev=prev_chunk,            # (B, W, H, A) previous action chunk
        d=d,                          # inference delay in controller steps
        s=s,                          # execution horizon in controller steps
        beta=5.0,                     # guidance weight clipping
    )  # -> (B, W, max_H, max_A)

References:
    Black et al. "Real-Time Execution of Action Chunking Flow Policies."
    NeurIPS 2025. https://arxiv.org/abs/2506.07339
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from einops import rearrange
import numpy as np

from crossformer.utils.mytyping import PRNGKey


# ---------------------------------------------------------------------------
# Soft mask (W) — Equation 5 from the paper
# ---------------------------------------------------------------------------


def build_soft_mask(H: int, d: int, s: int) -> jnp.ndarray:
    """Build soft inpainting mask W of shape (H,).

    Regions:
        [0, d)       → weight = 1.0   (frozen, will be executed)
        [d, H-s)     → exponential decay from 1 to 0 (intermediate)
        [H-s, H)     → weight = 0.0   (beyond previous chunk, freshly generated)

    Constraint: d <= s <= H - d  (from paper)

    Args:
        H: prediction horizon.
        d: inference delay (controller steps).
        s: execution horizon (controller steps).

    Returns:
        (H,) float array of guidance weights.
    """
    W = []
    for i in range(H):
        if i < d:
            w = 1.0
        elif i < H - s:
            c_i = (H - s - i) / (H - s - d + 1)
            w = (c_i * np.e ** c_i - 1) / (np.e - 1)
            w = float(np.clip(w, 0.0, 1.0))
        else:
            w = 0.0
        W.append(w)
    return jnp.array(W, dtype=jnp.float32)  # (H,)


# ---------------------------------------------------------------------------
# ΠGDM guidance term — Equation 2 from the paper
# ---------------------------------------------------------------------------


def _guidance_correction(
    velocity_fn,
    a_t: jnp.ndarray,
    time_val: float,
    y: jnp.ndarray,
    W_flat: jnp.ndarray,
    beta: float,
    dt: float,
) -> jnp.ndarray:
    """Compute ΠGDM guidance correction for one denoising step.

    Args:
        velocity_fn: callable (a_t, time_val) -> velocity, shape (B, W, H*A).
        a_t: (B, W, H*A) current noisy actions.
        time_val: float, current flow timestep tau in (0, 1].
        y: (B, W, H*A) target — right-padded previous chunk (a_prev flattened).
        W_flat: (H*A,) soft mask weights, broadcast over batch.
        beta: guidance weight clipping value.
        dt: step size = 1/n_steps.

    Returns:
        (B, W, H*A) guidance correction to add to velocity.
    """
    tau = time_val

    # r_tau^2 = (1-tau)^2 / (tau^2 + (1-tau)^2)  — Eq. 4
    r2 = (1.0 - tau) ** 2 / (tau ** 2 + (1.0 - tau) ** 2 + 1e-8)

    # Guidance weight scalar: min(beta, (1-tau) / (tau * r2))
    raw_weight = (1.0 - tau) / (tau * r2 + 1e-8)
    guide_weight = jnp.minimum(beta, raw_weight)

    # Ac1 = a_t + (1 - tau) * v(a_t)  — one-step denoised estimate, Eq. 3
    # We need VJP: (y - Ac1)^T diag(W) * d(Ac1)/d(a_t)
    # = VJP of Ac1 w.r.t a_t, evaluated at cotangent (y - Ac1) * W

    def ac1_fn(a):
        v = velocity_fn(a, time_val)
        return a + (1.0 - tau) * v  # (B, W, H*A)

    ac1, vjp_fn = jax.vjp(ac1_fn, a_t)

    # Weighted error: (y - Ac1) * W  — Eq. 2 numerator
    error = (y - ac1) * W_flat[None, None, :]  # (B, W, H*A)

    # VJP: g = error^T * d(Ac1)/d(a_t)
    (g,) = vjp_fn(error)  # (B, W, H*A)

    return guide_weight * g  # (B, W, H*A)


# ---------------------------------------------------------------------------
# RTC predict_action
# ---------------------------------------------------------------------------


def rtc_predict_action(
    bound,
    transformer_outputs: dict,
    rng: PRNGKey,
    dof_ids: jnp.ndarray,
    chunk_steps: jnp.ndarray,
    a_prev: jnp.ndarray,
    d: int,
    s: int,
    beta: float = 5.0,
    slot_pos=None,
    train: bool = False,
    guidance_tokens=None,
    guidance_mask=None,
    clip_pred: bool = True,
    max_action: float = 5.0,
) -> jnp.ndarray:
    """Predict next action chunk using RTC (inpainting via ΠGDM guidance).

    Generates the next chunk while incorporating previous chunk continuity.
    Freezes actions in [0, d) and softly guides [d, H-s) toward a_prev.

    Args:
        bound: bound XFlowHead (model.module.bind({"params": params})).
        transformer_outputs: dict with readout_key -> TokenGroup (B, W, N, E).
        rng: JAX PRNGKey.
        dof_ids: (B, max_A) int — MASK-padded DOF IDs.
        chunk_steps: (B, max_H) float — padded temporal positions.
        a_prev: (B, W, H_prev, max_A) previous action chunk.
                H_prev may be <= max_H; right-padded with zeros if shorter.
        d: inference delay in controller steps. Must satisfy d <= s.
        s: execution horizon in controller steps. Must satisfy d <= s <= H - d.
        beta: guidance weight clipping (default 5.0, per paper ablation).
        slot_pos: (B, max_A) float — ordinal slot positions (optional).
        train: passed to velocity network (should be False at inference).
        guidance_tokens: (B, G, E) optional extra conditioning tokens.
        guidance_mask: (B, G) int mask for classifier-free guidance.
        clip_pred: clip output to [-max_action, max_action].
        max_action: clipping bound.

    Returns:
        (B, W, max_H, max_A) predicted action chunk.
    """
    if hasattr(bound, "heads"):
        head = bound.heads["xflow"]
    else:
        head = bound
    module, variables = head.unbind()


    max_H = head.max_horizon
    max_A = head.max_dofs
    base_std = head.base_std
    flow_steps = head.flow_steps

    tokens = transformer_outputs[head.readout_key].tokens
    B, W = tokens.shape[:2]

    # -- Validate constraints ------------------------------------------------
    assert d >= 0, f"d must be >= 0, got {d}"
    assert s >= d, f"s must be >= d, got s={s}, d={d}"
    assert s <= max_H - d, f"s must be <= H-d={max_H - d}, got s={s}"

    # -- Build soft mask W ---------------------------------------------------
    W_flat = build_soft_mask(max_H, d, s)               # (max_H,)
    W_flat = jnp.tile(W_flat[:, None], (1, max_A))      # (max_H, max_A)
    W_flat = W_flat.reshape(-1)                          # (max_H * max_A,)

    # -- Prepare y: right-pad a_prev to (B, W, max_H, max_A) ----------------
    H_prev = a_prev.shape[2]
    if H_prev < max_H:
        pad = jnp.zeros((B, W, max_H - H_prev, max_A), dtype=a_prev.dtype)
        a_prev_padded = jnp.concatenate([a_prev, pad], axis=2)
    else:
        a_prev_padded = a_prev[:, :, :max_H, :]

    y = a_prev_padded.reshape(B, W, max_H * max_A)      # (B, W, max_H*max_A)

    # -- slot_pos default ----------------------------------------------------
    if slot_pos is None:
        slot_pos = jnp.broadcast_to(
            jnp.arange(max_A, dtype=jnp.float32),
            dof_ids.shape,
        )

    # -- Velocity function (wraps module.apply) -------------------------------
    def _velocity(a_t, time_val):
        """Call XFlowHead forward pass to get velocity field."""
        t = jnp.full((B, W, 1), time_val, dtype=a_t.dtype)
        return module.apply(
            variables,
            transformer_outputs,
            t,
            a_t,
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            slot_pos=slot_pos,
            train=train,
            guidance_tokens=guidance_tokens,
            guidance_mask=guidance_mask,
        )  # (B, W, max_H * max_A)

    # -- Euler integration with ΠGDM guidance --------------------------------
    dt = 1.0 / max(flow_steps, 1)

    rng, key = jax.random.split(rng)
    a_t = base_std * jax.random.normal(key, (B, W, max_H * max_A))

    def scan_fn(a_t, step):
        time_val = (step + 0.5) * dt

        # Base velocity
        velocity = _velocity(a_t, time_val)

        # ΠGDM guidance correction
        correction = _guidance_correction(
            velocity_fn=_velocity,
            a_t=a_t,
            time_val=time_val,
            y=y,
            W_flat=W_flat,
            beta=beta,
            dt=dt,
        )

        updated = a_t + dt * (velocity + correction)
        if clip_pred:
            updated = jnp.clip(updated, -max_action, max_action)
        return updated, ()

    steps = jnp.arange(flow_steps)
    a_t, _ = jax.lax.scan(scan_fn, a_t, steps)

    actions = rearrange(a_t, "b w (h a) -> b w h a", h=max_H, a=max_A)
    if clip_pred:
        actions = jnp.clip(actions, -max_action, max_action)

    return actions  # (B, W, max_H, max_A)


# ---------------------------------------------------------------------------
# Soft mask helper (numpy, for delay estimation outside JAX)
# ---------------------------------------------------------------------------


def compute_soft_mask_np(H: int, d: int, s: int) -> np.ndarray:
    """Numpy version of build_soft_mask for use outside JAX (e.g. logging)."""
    W = []
    for i in range(H):
        if i < d:
            w = 1.0
        elif i < H - s:
            c_i = (H - s - i) / (H - s - d + 1)
            w = float((c_i * np.e ** c_i - 1) / (np.e - 1))
            w = float(np.clip(w, 0.0, 1.0))
        else:
            w = 0.0
        W.append(w)
    return np.array(W, dtype=np.float32)
