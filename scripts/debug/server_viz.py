"""Forward-pass visualization: load data, run PolicyV2, rasterize actions, log to wandb.

Loads dataset samples (batch_size=1, mp=0), runs forward pass through
build_policy_v2 (with obs_pad + denorm wrappers), and renders robot
silhouettes via RastCallback.

Usage:
    uv run scripts/debug/server_viz.py \
        --path /path/to/checkpoint --task sweep \
        --rast-urdf xarm7_standalone.urdf --rast-mesh-dir assets \
        --rast-cams data/cam/over/HT.npz data/cam/side/HT.npz
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import jax
from jax.sharding import Mesh
import numpy as np
from rich import print
from rich.rule import Rule
import tyro

import crossformer.cn as cn
from crossformer.cn.dataset import DataSourceE
from crossformer.cn.dataset.dataset import Loader
from crossformer.cn.wab import Wandb
from crossformer.data.grain.loader import GrainDataFactory
from crossformer.embody import DOF
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.server import build_policy_v2, PolicyV2Config, TASKS
from crossformer.run.train_step import lookup_guide
from crossformer.run.wrappers import BodyPartGroupWrapper
from crossformer.utils.callbacks.rast import RastCallback
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer
from crossformer.utils.spec import spec
import wandb

JOINT_NAMES = tuple(f"j{i}" for i in range(7))
JOINT_IDS = tuple(DOF[name] for name in JOINT_NAMES)
JOINT_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(JOINT_IDS)}


@dataclass
class Config:
    path: str = ""  # checkpoint path (required for local, needed for GT denorm with remote)
    task: str = "sweep"  # task name (must match TASKS dict)
    step: int | None = None
    name: str = ""

    # remote — if set, use websocket client instead of local policy
    host: str | None = None
    port: int = 8001

    # serve — if set, wrap as a policy server on this port
    serve_port: int | None = None

    mix: str = "xgym_sweep_single"
    n_samples: int = 8
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")
    no_guide: bool = False  # disable guide_input for A/B testing

    # rast
    rast_urdf: Path | None = Path("xarm7_standalone.urdf")
    rast_mesh_dir: Path | None = Path("assets")
    rast_cams: tuple[Path, ...] = (
        Path("data/cam/over/HT.npz"),
        Path("data/cam/side/HT.npz"),
        Path("data/cam/low/HT.npz"),
    )
    rast_width: int = 256
    rast_height: int = 256

    # wandb
    wandb: Wandb = field(default_factory=lambda: Wandb(project="crossformer-server-viz"))


def shard_batch(batch, mesh):
    """Identity shard for batch_size=1 (no partitioning needed)."""
    return batch


class ReplayPolicy:
    """Wraps a policy + dataset iterator. Ignores incoming obs, returns dataset obs + predicted actions."""

    def __init__(self, policy, dsit, denorm, ds_name, guide_keys, no_guide):
        self.policy = policy
        self.dsit = dsit
        self.denorm = denorm
        self.ds_name = ds_name
        self.guide_keys = guide_keys
        self.no_guide = no_guide

    def step(self, obs: dict) -> dict:
        batch = jax.device_get(next(self.dsit))
        act = np.asarray(batch["act"]["base"], dtype=np.float32)
        dof_ids = np.asarray(batch["act"]["id"])
        task = batch.get("task", {"pad_mask_dict": {}})
        H = act.shape[1]
        chunk_steps = np.arange(H, dtype=np.float32)

        guide_input = None
        if not self.no_guide:
            with contextlib.suppress(KeyError):
                guide_input = lookup_guide(batch, self.guide_keys)

        self.policy.reset()
        payload = {
            "observation": batch["observation"],
            "task": task,
            "dof_ids": dof_ids[0],
            "chunk_steps": chunk_steps,
            "guide_input": guide_input,
        }
        payload = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, payload)
        obs_keys = sorted(payload["observation"].keys())
        has_pmd = "pad_mask_dict" in payload["observation"]
        print(f"[CLIENT] sending obs keys: {obs_keys}")
        print(f"[CLIENT] has pad_mask_dict: {has_pmd}")
        if has_pmd:
            pmd = payload["observation"]["pad_mask_dict"]
            print(f"[CLIENT] pad_mask_dict values: {{{', '.join(f'{k}: {v}' for k, v in pmd.items())}}}")
        result = self.policy.step(payload)

        pred_ids = np.asarray(result.get("dof_ids", dof_ids[0]))
        gt_ids = dof_ids[0]
        gt_joints = slots_to_joints(act[0], gt_ids)
        gt_joints = np.stack(
            [
                self.denorm.denormalize_slot(gt_joints[t], np.array(JOINT_IDS), self.ds_name)
                for t in range(len(gt_joints))
            ]
        )

        return {
            "observation": batch["observation"],
            "actions": result["actions"],
            "gt_actions": gt_joints,
            "dof_ids": pred_ids,
        }

    def reset(self):
        pass


def slots_to_joints(arr: np.ndarray, dof_ids: np.ndarray) -> np.ndarray:
    """Remap (H, A) slot-ordered actions to (H, 7) j0..j6 canonical order."""
    out = np.zeros((*arr.shape[:-1], 7), dtype=np.float32)
    for src, dof_id in enumerate(dof_ids):
        dst = JOINT_ID_TO_IDX.get(int(dof_id))
        if dst is not None:
            out[..., dst] = arr[..., src]
    return out


def action_joints(actions, dof_ids: np.ndarray) -> np.ndarray:
    """Extract joint actions from flat or grouped policy output."""
    if isinstance(actions, dict):
        if "joints" not in actions:
            keys = ", ".join(sorted(actions))
            raise KeyError(f"Expected grouped actions to contain 'joints', got: {keys}")
        return np.asarray(actions["joints"], dtype=np.float32)
    return slots_to_joints(np.asarray(actions), dof_ids)


def main(cfg: Config):
    mesh = Mesh(jax.devices(), axis_names="batch")

    # -- wandb --
    cfg.wandb.initialize(cfg, name=f"server-viz-{cfg.task}")

    # -- data (batch_size=1, mp=0) --
    train_cfg = cn.Train(
        data=cn.Dataset(
            mix=DataSourceE[cfg.mix],
            loader=Loader(use_grain=True, global_batch_size=1),
        ),
        seed=42,
        verbosity=0,
    )
    dataset = GrainDataFactory(mp=0, shuffle=False).make(
        train_cfg,
        shard_fn=partial(shard_batch, mesh=mesh),
        train=True,
    )
    dsit = iter(dataset.dataset)

    # -- policy (local or remote) --
    if cfg.host is not None:
        from webpolicy.client import Client

        policy = Client(host=cfg.host, port=cfg.port)
    else:
        policy = build_policy_v2(
            PolicyV2Config(
                path=cfg.path,
                task=cfg.task,
                step=cfg.step,
                obs_pad=True,
                resize=False,
                proprio_norm=False,
                ensemble=False,
                denorm=True,
                warmup=False,
            )
        )
        # server_viz rasterizes canonical joints from flat slot actions using
        # dof_ids. Strip the outer grouping wrapper added by build_policy_v2.
        if isinstance(policy, BodyPartGroupWrapper):
            policy = policy.inner

    ds_name = TASKS[cfg.task]["dataset_name"]
    # GT denorm needs dataset stats — load from checkpoint even when using remote policy
    if hasattr(policy, "unwrapped"):
        stats = policy.unwrapped().model.dataset_statistics
    else:
        model = CrossFormerModel.load_pretrained(cfg.path, step=cfg.step)
        stats = model.dataset_statistics
    denorm = ActionBatchDenormalizer(stats)

    # -- rast callback (lazy init — needs CUDA) --
    rast_cb = None
    if cfg.rast_urdf is not None:
        rast_cb = RastCallback(
            urdf=cfg.rast_urdf,
            cams=list(cfg.rast_cams) if cfg.rast_cams else None,
            mesh_dir=cfg.rast_mesh_dir,
            width=cfg.rast_width,
            height=cfg.rast_height,
        )

    print(Rule("server viz"))
    print(
        {
            "path": cfg.path,
            "task": cfg.task,
            "mix": cfg.mix,
            "n_samples": cfg.n_samples,
        }
    )

    # -- serve mode: wrap as a policy server --
    if cfg.serve_port is not None:
        from webpolicy.server import Server

        replay = ReplayPolicy(policy, dsit, denorm, ds_name, cfg.guide_keys, cfg.no_guide)
        server = Server(replay, host="0.0.0.0", port=cfg.serve_port)
        print(f"serving replay policy on 0.0.0.0:{cfg.serve_port}")
        server.serve()
        return

    for i in range(cfg.n_samples):
        batch = jax.device_get(next(dsit))
        act = np.asarray(batch["act"]["base"], dtype=np.float32)  # (1, H, A)
        dof_ids = np.asarray(batch["act"]["id"])  # (1, A)
        task = batch.get("task", {"pad_mask_dict": {}})
        H = act.shape[1]
        chunk_steps = np.arange(H, dtype=np.float32)

        guide_input = None
        if not cfg.no_guide:
            with contextlib.suppress(KeyError):
                guide_input = lookup_guide(batch, cfg.guide_keys)

        # -- forward pass via policy.step --
        policy.reset()
        payload = {
            "observation": batch["observation"],
            "task": task,
            "dof_ids": dof_ids[0],
            "chunk_steps": chunk_steps,
            "guide_input": guide_input,
        }
        # webpolicy Client uses msgpack which can't serialize JAX arrays
        payload = jax.tree.map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, payload)
        obs_keys = sorted(payload["observation"].keys())
        has_pmd = "pad_mask_dict" in payload["observation"]
        print(f"[CLIENT] sending obs keys: {obs_keys}")
        print(f"[CLIENT] has pad_mask_dict: {has_pmd}")
        if has_pmd:
            pmd = payload["observation"]["pad_mask_dict"]
            print(f"[CLIENT] pad_mask_dict values: {{{', '.join(f'{k}: {v}' for k, v in pmd.items())}}}")
        result = policy.step(payload)
        pred_actions = result["actions"]  # (H, A) — already denormalized

        if i == 0:
            print(spec(batch["observation"]))

        # remap to j0..j6 canonical order for rendering
        pred_ids = np.asarray(result.get("dof_ids", dof_ids[0]))
        gt_ids = dof_ids[0]
        pred_joints = action_joints(pred_actions, pred_ids)
        gt_joints = slots_to_joints(act[0], gt_ids)
        gt_joints = np.stack(
            [denorm.denormalize_slot(gt_joints[t], np.array(JOINT_IDS), ds_name) for t in range(len(gt_joints))]
        )

        # render
        log = {}
        if rast_cb is not None:
            try:
                pred_frames = rast_cb.render_trajectory(pred_joints)
                gt_frames = rast_cb.render_trajectory(gt_joints)
                for ci, (pf, gf) in enumerate(zip(pred_frames, gt_frames)):
                    log[f"model/sample_{i}/cam_{ci}"] = wandb.Image(pf)
                    log[f"gt/sample_{i}/cam_{ci}"] = wandb.Image(gf)
            except RuntimeError as e:
                print(f"[yellow]rast skipped (no CUDA): {e}[/]")
                rast_cb = None
        cfg.wandb.log(log, step=i)
        pred_shape = pred_joints.shape if isinstance(pred_actions, dict) else pred_actions.shape
        print(f"sample {i}: pred {pred_shape} gt {gt_joints.shape}")

    if cfg.wandb.use:
        wandb.finish()
    print(Rule("done"))


if __name__ == "__main__":
    main(tyro.cli(Config))
