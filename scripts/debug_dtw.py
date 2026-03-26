from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from crossformer import cn
from crossformer.data.grain.loader import GrainDataFactory
from crossformer.utils.dtw import compute_dtw_matrix_single, compute_dtw_path
from crossformer.viz.dtw_plot import render_dtw_alignment_figure


def inspect_batch(batch: dict, prefix: str = "") -> None:
    """Recursively print shapes of all arrays in a nested dict batch."""
    for k, v in sorted(batch.items()):
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            inspect_batch(v, key)
        elif hasattr(v, "shape"):
            print(f"  {key:50s} shape={shape:20s} dtype={v.dtype}")
        else:
            print(f"  {key:50s} type={type(v).__name__}")


def debug_dtw(cfg: cn.Train):
    # ── 1. Dataset setup ──────────────────────────────────────────────
    print(f"[init] Loading dataset: {cfg.data.mix}")
    mesh = jax.sharding.Mesh(jax.devices(), axis_names="batch")

    def shard_fn(batch):
        return jax.experimental.multihost_utils.host_local_array_to_global_array(
            batch, mesh, jax.sharding.PartitionSpec("batch")
        )

    data_factory = GrainDataFactory()
    dataset = data_factory.make(cfg, shard_fn=shard_fn, train=True)
    dsit = iter(dataset.dataset)

    # ── 2. Probe a single batch to show the shape mismatch ────────────
    probe = jax.device_get(next(dsit))
    print("\n[probe] Batch structure:")
    inspect_batch(probe)

    obs_joints = np.asarray(probe["observation"]["proprio_joints"], dtype=np.float32)
    act_joints = np.asarray(probe["action"]["joints"], dtype=np.float32)
    print(f"\n[probe] observation.proprio_joints : {obs_joints.shape}")
    print(f"[probe] action.joints              : {act_joints.shape}")

    # Determine action horizon from the action tensor
    # act_joints is expected to be [Batch, Horizon, DOF]
    horizon = act_joints.shape[1]
    dof = act_joints.shape[-1]
    sample_idx = 0

    print(f"\n[info] Action horizon = {horizon}, DOF = {dof}")
    print(f"[info] Observation window = {obs_joints.shape[1]} (single step — must buffer)")

    # ── 3. Buffer N observation steps to build a human trajectory ─────
    # We already consumed 1 batch; use its observation as the first step.
    n_steps = horizon  # match the action sequence length
    print(f"\n[buffer] Collecting {n_steps} consecutive observation steps...")

    # Start the buffer with the observation from the probe batch
    human_buf = [obs_joints[sample_idx, 0, :dof]]  # [DOF]

    for i in range(1, n_steps):
        batch_i = jax.device_get(next(dsit))
        obs_i = np.asarray(batch_i["observation"]["proprio_joints"], dtype=np.float32)
        human_buf.append(obs_i[sample_idx, 0, :dof])

    human_traj = np.stack(human_buf, axis=0)  # [N, DOF]

    # Robot trajectory: action horizon from the *last* batch (most recent context)
    robot_traj = act_joints[sample_idx, :, :dof]  # [Horizon, DOF]

    print(f"\n[result] Human trajectory  (buffered obs) : {human_traj.shape}")
    print(f"[result] Robot trajectory  (action horizon): {robot_traj.shape}")
    assert human_traj.shape == robot_traj.shape, f"Shape mismatch: human={human_traj.shape} vs robot={robot_traj.shape}"

    # ── 4. Compute DTW ────────────────────────────────────────────────
    print("\n[dtw] Computing DTW alignment...")
    band_r = min(15, n_steps)

    D_matrix = np.asarray(
        compute_dtw_matrix_single(
            jnp.array(human_traj),
            jnp.array(robot_traj),
            R=band_r,
        )
    )
    path = compute_dtw_path(D_matrix)
    print(f"[dtw] Cost matrix shape : {D_matrix.shape}")
    print(f"[dtw] Warping path len  : {len(path)}")
    print(f"[dtw] Final cost        : {D_matrix[-1, -1]:.4f}")

    # ── 5. Render per-joint plots ─────────────────────────────────────
    out_dir = Path("debug_dtw_out")
    out_dir.mkdir(exist_ok=True)

    joint_idx = 1  # shoulder joint — usually most dynamic
    human_1d = human_traj[:, joint_idx]
    robot_1d = robot_traj[:, joint_idx]

    rgb = render_dtw_alignment_figure(
        human_1d,
        robot_1d,
        path,
        title_a=f"Human Obs (joint {joint_idx})",
        title_b=f"Robot Act (joint {joint_idx})",
    )

    out_path = out_dir / "dtw_alignment.png"
    from PIL import Image

    Image.fromarray(rgb).save(out_path)
    print(f"\n[done] Saved alignment plot to {out_path.resolve()}")


if __name__ == "__main__":
    debug_dtw(tyro.cli(cn.Train))
