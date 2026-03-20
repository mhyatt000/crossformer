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

from crossformer.embody import ids, MASK_ID, VOCAB_SIZE
from crossformer.model.components.diffusion import FourierFeatures

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
    "bimanual_joint": ids(
        "j0",
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
        "gripper",
        "j0",
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
        "gripper",
    ),
    "mano": ids(*[f"mano_{i}" for i in range(7)]),
    "k3ds": ids(*[f"k3d_{i}" for i in range(84)]),
}

# ---------------------------------------------------------------------------
# Chunk step helpers
# ---------------------------------------------------------------------------

# Sentinel for padded positions (Fourier-encoded but masked in attention)
CHUNK_PAD = -1.0
SLOT_PAD = -1.0


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


def pad_slot_positions(slots: tuple[float, ...], max_a: int) -> tuple[float, ...]:
    """Pad slot positions to max_a with SLOT_PAD sentinel."""
    return slots + (SLOT_PAD,) * (max_a - len(slots))


def build_query_mask(
    chunk_steps: Array,
    dof_ids: Array,
    slot_positions: Array | None = None,
) -> Array:
    """Build (batch, max_H * max_A) bool mask from padded inputs.

    Args:
        chunk_steps: (batch, max_H) float — CHUNK_PAD marks padding.
        dof_ids: (batch, max_A) int — MASK_ID marks padding.
        slot_positions: (batch, max_A) float — SLOT_PAD marks padding (optional).

    Returns:
        (batch, max_H * max_A) bool mask, True for valid queries.
    """
    chunk_valid = chunk_steps != CHUNK_PAD  # (B, max_H)
    dof_valid = dof_ids != MASK_ID  # (B, max_A)
    if slot_positions is not None:
        dof_valid = dof_valid & (slot_positions != SLOT_PAD)
    mask = chunk_valid[:, :, None] & dof_valid[:, None, :]  # (B, max_H, max_A)
    return rearrange(mask, "b h a -> b (h a)")


# ---------------------------------------------------------------------------
# Factored query encoding
# ---------------------------------------------------------------------------


class FactoredQueryEncoding(nn.Module):
    """Factored query encoding: Fourier chunk + learned DOF + Fourier slot.

    Three axes, three embeddings summed per query token:
      chunk (ordinal, temporal)  — which timestep in the action horizon
      dof   (categorical, joint) — what kind of joint (shared across embodiments)
      slot  (ordinal, position)  — where in the action vector (breaks symmetry
            for repeated DOFs, e.g. bimanual arms share DOF IDs but differ in slot)

    Output: (B, H*A, D) query encodings via outer sum then flatten.
    Accepts per-sample inputs for single-kernel cross-embodiment training.
    """

    num_channels: int = 256
    dof_vocab_size: int = VOCAB_SIZE
    chunk_fourier_dim: int = 32
    slot_fourier_dim: int = 32

    @nn.compact
    def __call__(
        self,
        chunk_steps: Array,
        dof_ids: Array,
        slot_pos: Array,
    ) -> Array:
        """
        Args:
            chunk_steps: (batch, max_H) float — temporal positions.
            dof_ids: (batch, max_A) int — DOF vocab IDs.
            slot_pos: (batch, max_A) float — ordinal position in action vector.

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

        # Slot: (B, max_A, 1) → Fourier → Dense → (B, max_A, D)
        slot_emb = FourierFeatures(
            self.slot_fourier_dim,
            learnable=True,
            name="slot_ff",
        )(slot_pos[..., None])
        slot_emb = nn.Dense(D, name="slot_proj")(slot_emb)

        # Sum DOF + slot per action position, then outer sum with chunk
        act_emb = dof_emb + slot_emb  # (B, max_A, D)
        pos_q = chunk_emb[:, :, None, :] + act_emb[:, None, :, :]
        return rearrange(pos_q, "b h a d -> b (h a) d")
