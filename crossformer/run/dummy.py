from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import jax.numpy as jnp

from crossformer.embody import Embodiment, MASK_ID

EmbodimentLike: TypeAlias = str | Embodiment


def resolve_embodiment(embodiment: EmbodimentLike) -> Embodiment:
    """Resolve an embodiment name or object."""
    if isinstance(embodiment, Embodiment):
        return embodiment
    try:
        return Embodiment.REGISTRY[embodiment]
    except KeyError as e:
        names = ", ".join(sorted(Embodiment.REGISTRY))
        raise ValueError(f"Unknown embodiment {embodiment!r}. Available: {names}") from e


def resolve_embodiments(embodiment: EmbodimentLike | Sequence[EmbodimentLike]) -> tuple[Embodiment, ...]:
    """Resolve one or more embodiments."""
    if isinstance(embodiment, str | Embodiment):
        return (resolve_embodiment(embodiment),)
    out = tuple(resolve_embodiment(e) for e in embodiment)
    if not out:
        raise ValueError("At least one embodiment is required")
    return out


def make_chunk_steps(batch: dict) -> jnp.ndarray:
    """Return chunk positions with shape (B, H)."""
    actions = batch["act"]["base"]
    return jnp.tile(jnp.arange(actions.shape[2], dtype=jnp.float32)[None], (actions.shape[0], 1))


def make_fake_batch(
    *,
    batch_size: int = 2,
    obs_horizon: int = 4,
    action_horizon: int = 16,
    num_views: int = 3,
    embodiment: EmbodimentLike | Sequence[EmbodimentLike] = "cart_gripper",
    proprio_key: str = "proprio",
    image_hw: tuple[int, int] = (16, 16),
) -> dict:
    """Create a deterministic bundled-action batch for smoke tests.

    Shapes:
        observation[proprio_key]: (B, W, max_A)
        observation["image"]:    (B, W, V, H, W, C)
        act["base"]:             (B, W, action_H, max_A)
        act["id"]:               (B, max_A), MASK-padded with 0
    """
    if batch_size <= 0 or obs_horizon <= 0 or action_horizon <= 0:
        raise ValueError("batch_size, obs_horizon, and action_horizon must be positive")
    if num_views <= 0:
        raise ValueError("num_views must be positive")

    embodiments = resolve_embodiments(embodiment)
    sample_embodiments = tuple(embodiments[i % len(embodiments)] for i in range(batch_size))
    max_a = max(e.action_dim for e in embodiments)

    actions = jnp.zeros((batch_size, obs_horizon, action_horizon, max_a), dtype=jnp.float32)
    proprio = jnp.zeros((batch_size, obs_horizon, max_a), dtype=jnp.float32)
    dof_ids = jnp.full((batch_size, max_a), MASK_ID, dtype=jnp.int32)
    views = jnp.zeros((batch_size, max_a), dtype=jnp.int32)
    act_mask = jnp.zeros((batch_size, max_a), dtype=bool)

    h_pos = jnp.arange(action_horizon, dtype=jnp.float32)[:, None]
    w_pos = jnp.arange(obs_horizon, dtype=jnp.float32)[:, None]
    for b, emb in enumerate(sample_embodiments):
        ids = jnp.asarray(emb.dof_ids, dtype=jnp.int32)
        part_views = jnp.asarray([p.view for p in emb.expanded for _ in p.dof_ids], dtype=jnp.int32)
        slots = jnp.arange(emb.action_dim, dtype=jnp.float32)[None, :]

        base = 0.01 * (b + 1) + 0.1 * h_pos + 0.001 * slots
        obs = 0.01 * (b + 1) + 0.01 * w_pos + 0.001 * slots
        actions = actions.at[b, :, :, : emb.action_dim].set(base[None])
        proprio = proprio.at[b, :, : emb.action_dim].set(obs)
        dof_ids = dof_ids.at[b, : emb.action_dim].set(ids)
        views = views.at[b, : emb.action_dim].set(part_views)
        act_mask = act_mask.at[b, : emb.action_dim].set(True)

    image_h, image_w = image_hw
    image = jnp.zeros((batch_size, obs_horizon, num_views, image_h, image_w, 3), dtype=jnp.float32)
    timestep_mask = jnp.ones((batch_size, obs_horizon), dtype=bool)
    view_mask = jnp.ones((batch_size, obs_horizon, num_views), dtype=bool)
    emb_mask = {
        emb.name: jnp.asarray([sample.name == emb.name for sample in sample_embodiments], dtype=bool)
        for emb in embodiments
    }

    return {
        "observation": {
            proprio_key: proprio,
            "image": image,
            "view_mask": view_mask,
            "timestep_pad_mask": timestep_mask,
            "pad_mask_dict": {
                proprio_key: timestep_mask,
                "image": timestep_mask,
            },
        },
        "task": {"pad_mask_dict": {}},
        "state": {
            "base": actions[:, :, 0, :],
            "id": dof_ids,
            "view": views,
        },
        "act": {
            "base": actions,
            "id": dof_ids,
            "view": views,
        },
        "mask": {
            "act": act_mask,
            "state": {"base": act_mask},
            "embodiment": emb_mask,
        },
    }
