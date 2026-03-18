"""DOF vocabulary and factored query encoding for cross-embodiment action prediction.

Each degree of freedom (DOF) gets a unique integer ID used as an nn.Embed index.
Embodiments are recipes — ordered tuples of DOF IDs defining a robot's action space.
Shared DOFs (e.g. arm joints) reuse the same embedding across embodiments.
"""

from __future__ import annotations

from einops import rearrange
import flax.linen as nn
import jax.numpy as jnp

from crossformer.model.components.diffusion import FourierFeatures

# ---------------------------------------------------------------------------
# DOF vocabulary
# ---------------------------------------------------------------------------

DOF: dict[str, int] = {
    # 7-DOF arm joints (shared across xarm, franka, kuka, etc.)
    "j0": 0,
    "j1": 1,
    "j2": 2,
    "j3": 3,
    "j4": 4,
    "j5": 5,
    "j6": 6,
    # End-effector
    "gripper": 7,
    # Cartesian end-effector pose
    "ee_x": 8,
    "ee_y": 9,
    "ee_z": 10,
    "ee_rx": 11,
    "ee_ry": 12,
    "ee_rz": 13,
    # Dexterous hand (16 DOF — covers allegro, ruka, etc.)
    **{f"hand_j{i}": 14 + i for i in range(16)},
    # Mobile base
    "base_vx": 30,
    "base_vy": 31,
    "base_wz": 32,
}

VOCAB_SIZE = 128  # embedding table size (room to grow)


def ids(*names: str) -> tuple[int, ...]:
    """Look up DOF names → integer IDs."""
    return tuple(DOF[n] for n in names)


# ---------------------------------------------------------------------------
# Embodiment recipes
# ---------------------------------------------------------------------------

EMBODIMENTS: dict[str, tuple[int, ...]] = {
    "xarm_gripper": ids("j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"),
    "xarm": ids("j0", "j1", "j2", "j3", "j4", "j5", "j6"),
    "franka_gripper": ids("j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"),
    "ur5_gripper": ids("j0", "j1", "j2", "j3", "j4", "j5", "gripper"),
    "cartesian_pose": ids("ee_x", "ee_y", "ee_z", "ee_rx", "ee_ry", "ee_rz"),
    "cartesian_pos": ids("ee_x", "ee_y", "ee_z"),
    "cart_pose_gripper": ids(
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_rx",
        "ee_ry",
        "ee_rz",
        "gripper",
    ),
    "xarm_ruka": ids(
        "j0",
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
        *[f"hand_j{i}" for i in range(11)],
    ),
    "pose_ruka": ids(
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_rx",
        "ee_ry",
        "ee_rz",
        *[f"hand_j{i}" for i in range(11)],
    ),
}

# ---------------------------------------------------------------------------
# Chunk step helpers
# ---------------------------------------------------------------------------


def chunk_range(horizon: int) -> tuple[float, ...]:
    """Dense chunk steps: (0., 1., ..., horizon-1.)."""
    return tuple(float(i) for i in range(horizon))


def chunk_strided(end: int, stride: int) -> tuple[float, ...]:
    """Strided chunk steps: (0., stride, 2*stride, ...)."""
    return tuple(float(i) for i in range(0, end, stride))


# ---------------------------------------------------------------------------
# Factored query encoding
# ---------------------------------------------------------------------------


class FactoredQueryEncoding(nn.Module):
    """Factored query positional encoding: Fourier chunk + learned DOF.

    Chunks are ordinal (temporal) → Fourier features + projection.
    DOFs are categorical (joint identity) → nn.Embed lookup.
    Output: outer sum of chunk and DOF embeddings, flattened to (H*A, D).
    """

    chunk_steps: tuple[float, ...]
    dof_ids: tuple[int, ...]
    num_channels: int = 256
    dof_vocab_size: int = VOCAB_SIZE
    chunk_fourier_dim: int = 32

    @nn.compact
    def __call__(self, batch_size: int | None = None):
        D = self.num_channels

        # Chunk: Fourier features on continuous step values
        steps = jnp.array(self.chunk_steps, dtype=jnp.float32)[:, None]  # (H, 1)
        chunk_emb = FourierFeatures(
            self.chunk_fourier_dim,
            learnable=True,
            name="chunk_ff",
        )(steps)
        chunk_emb = nn.Dense(D, name="chunk_proj")(chunk_emb)  # (H, D)

        # DOF: learned vocabulary embedding
        dof_emb = nn.Embed(self.dof_vocab_size, D, name="dof_embed")(
            jnp.array(self.dof_ids, dtype=jnp.int32),
        )  # (A, D)

        # Outer sum → flatten: (H, A, D) → (H*A, D)
        pos_q = chunk_emb[:, None, :] + dof_emb[None, :, :]
        pos_q = rearrange(pos_q, "h a d -> (h a) d")

        if batch_size is not None:
            pos_q = jnp.broadcast_to(pos_q[None], (batch_size, *pos_q.shape))
        return pos_q
