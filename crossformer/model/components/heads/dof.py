"""DOF vocabulary and factored query encoding for cross-embodiment action prediction.

Each degree of freedom (DOF) gets a unique integer ID used as an nn.Embed index.
Embodiments are recipes — ordered tuples of DOF IDs defining a robot's action space.
Shared DOFs (e.g. arm joints) reuse the same embedding across embodiments.

MASK (id=0) is a reserved padding token. Pad shorter embodiments and horizons
to max size, then mask attention for padded queries.
"""

from __future__ import annotations

from einops import rearrange
import flax.linen as nn
from jax import Array

from crossformer.model.components.diffusion import FourierFeatures

# ---------------------------------------------------------------------------
# DOF vocabulary
# ---------------------------------------------------------------------------

MASK_ID = 0  # reserved padding token

DOF: dict[str, int] = {
    # Reserved
    "MASK": MASK_ID,
    # 7-DOF arm joints (shared across xarm, franka, kuka, etc.)
    "j0": 1,
    "j1": 2,
    "j2": 3,
    "j3": 4,
    "j4": 5,
    "j5": 6,
    "j6": 7,
    # End-effector
    "gripper": 8,
    # Cartesian end-effector pose
    "ee_x": 9,
    "ee_y": 10,
    "ee_z": 11,
    "ee_rx": 12,
    "ee_ry": 13,
    "ee_rz": 14,
    # Dexterous hand (16 DOF — covers allegro, ruka, etc.)
    **{f"hand_j{i}": 15 + i for i in range(16)},
    # Mobile base
    "base_vx": 31,
    "base_vy": 32,
    "base_wz": 33,
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

# Sentinel for padded chunk positions (Fourier-encoded but masked in attention)
CHUNK_PAD = -1.0


def chunk_range(horizon: int) -> tuple[float, ...]:
    """Dense chunk steps: (0., 1., ..., horizon-1.)."""
    return tuple(float(i) for i in range(horizon))


def chunk_strided(end: int, stride: int) -> tuple[float, ...]:
    """Strided chunk steps: (0., stride, 2*stride, ...)."""
    return tuple(float(i) for i in range(0, end, stride))


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def pad_dof_ids(dof_ids: tuple[int, ...], max_a: int) -> tuple[int, ...]:
    """Pad DOF IDs to max_a with MASK_ID."""
    return dof_ids + (MASK_ID,) * (max_a - len(dof_ids))


def pad_chunk_steps(steps: tuple[float, ...], max_h: int) -> tuple[float, ...]:
    """Pad chunk steps to max_h with CHUNK_PAD sentinel."""
    return steps + (CHUNK_PAD,) * (max_h - len(steps))


def build_query_mask(
    chunk_steps: Array,
    dof_ids: Array,
) -> Array:
    """Build (batch, max_H * max_A) bool mask from padded inputs.

    Args:
        chunk_steps: (batch, max_H) float — CHUNK_PAD marks padding.
        dof_ids: (batch, max_A) int — MASK_ID marks padding.

    Returns:
        (batch, max_H * max_A) bool mask, True for valid queries.
    """
    chunk_valid = chunk_steps != CHUNK_PAD  # (B, max_H)
    dof_valid = dof_ids != MASK_ID  # (B, max_A)
    mask = chunk_valid[:, :, None] & dof_valid[:, None, :]  # (B, max_H, max_A)
    return rearrange(mask, "b h a -> b (h a)")


# ---------------------------------------------------------------------------
# Factored query encoding
# ---------------------------------------------------------------------------


class FactoredQueryEncoding(nn.Module):
    """Factored query positional encoding: Fourier chunk + learned DOF.

    Chunks are ordinal (temporal) → Fourier features + projection.
    DOFs are categorical (joint identity) → nn.Embed lookup.
    Output: outer sum of chunk and DOF embeddings, flattened to (B, H*A, D).

    Accepts per-sample inputs for single-kernel training across embodiments.
    MASK DOF and padded chunk positions are encoded but masked in attention.
    """

    num_channels: int = 256
    dof_vocab_size: int = VOCAB_SIZE
    chunk_fourier_dim: int = 32

    @nn.compact
    def __call__(self, chunk_steps: Array, dof_ids: Array) -> Array:
        """
        Args:
            chunk_steps: (batch, max_H) float — temporal positions.
            dof_ids: (batch, max_A) int — DOF vocab IDs.

        Returns:
            (batch, max_H * max_A, D) query encodings.
        """
        D = self.num_channels

        # Chunk: (B, max_H, 1) → Fourier → Dense → (B, max_H, D)
        chunk_emb = FourierFeatures(
            self.chunk_fourier_dim,
            learnable=True,
            name="chunk_ff",
        )(chunk_steps[..., None])
        chunk_emb = nn.Dense(D, name="chunk_proj")(chunk_emb)

        # DOF: (B, max_A) → Embed → (B, max_A, D)
        dof_emb = nn.Embed(self.dof_vocab_size, D, name="dof_embed")(dof_ids)

        # Outer sum → flatten: (B, max_H, max_A, D) → (B, max_H*max_A, D)
        pos_q = chunk_emb[:, :, None, :] + dof_emb[:, None, :, :]
        return rearrange(pos_q, "b h a d -> b (h a) d")
