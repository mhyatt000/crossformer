"""Forward-pass visualization: load data, run CorePolicy, rasterize actions, log to wandb.

Loads dataset samples (batch_size=1, mp=0), runs forward pass through
build_policy_v2 (no preprocessing — grain data is already normalized),
denormalizes predicted joint actions, and renders robot silhouettes via
RastCallback.

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
import jax.numpy as jnp
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
from crossformer.run.server import build_policy_v2, PolicyV2Config, TASKS
from crossformer.run.train_step import lookup_guide
from crossformer.utils.callbacks.rast import RastCallback
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer
from crossformer.utils.spec import spec
import wandb

JOINT_NAMES = tuple(f"j{i}" for i in range(7))
JOINT_IDS = tuple(DOF[name] for name in JOINT_NAMES)
JOINT_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(JOINT_IDS)}


@dataclass
class Config:
    path: str
    task: str
    step: int | None = None
    name: str = ""

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


def fill_missing_obs(obs: dict, example_obs: dict) -> dict:
    """Fill missing observation keys from example_batch (zeros) and mask them out."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in obs.items()}
    pad = out.get("pad_mask_dict", {})

    for key, example_value in example_obs.items():
        if key == "pad_mask_dict":
            for pk, pv in example_value.items():
                if pk not in pad:
                    pad[pk] = np.zeros_like(np.asarray(pv))  # False — masked out
            out["pad_mask_dict"] = pad
        elif key not in out:
            out[key] = np.zeros_like(np.asarray(example_value))
    return out


def slots_to_joints(arr: np.ndarray, dof_ids: np.ndarray) -> np.ndarray:
    """Remap (H, A) slot-ordered actions to (H, 7) j0..j6 canonical order."""
    out = np.zeros((*arr.shape[:-1], 7), dtype=np.float32)
    for src, dof_id in enumerate(dof_ids):
        dst = JOINT_ID_TO_IDX.get(int(dof_id))
        if dst is not None:
            out[..., dst] = arr[..., src]
    return out


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
    dataset = GrainDataFactory(mp=0).make(
        train_cfg,
        shard_fn=partial(shard_batch, mesh=mesh),
        train=True,
    )
    dsit = iter(dataset.dataset)

    # -- policy (bare CorePolicy — no preprocessing, no ensembler) --
    policy = build_policy_v2(
        PolicyV2Config(
            path=cfg.path,
            task=cfg.task,
            step=cfg.step,
            resize=False,
            proprio_norm=False,
            ensemble=False,
            denorm=False,
            warmup=False,
        )
    )

    example_obs = jax.device_get(policy.model.example_batch["observation"])
    ds_name = TASKS[cfg.task]["dataset_name"]
    denorm = ActionBatchDenormalizer(policy.model.dataset_statistics)

    print(spec(example_obs))

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

    for i in range(cfg.n_samples):
        batch = jax.device_get(next(dsit))
        act = np.asarray(batch["act"]["base"], dtype=np.float32)  # (1, H, A)
        dof_ids = np.asarray(batch["act"]["id"])  # (1, A)
        task = batch.get("task", {"pad_mask_dict": {}})
        H = act.shape[1]
        chunk_steps = np.arange(H, dtype=np.float32)

        policy.reset()
        policy.task = task

        obs = fill_missing_obs(batch["observation"], example_obs)
        if i == 0:
            print(spec(obs))

        # forward pass — call transformer + head directly (like xflow.py) for guide_input
        obs_jax = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), obs)
        task_jax = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x)), task)
        pad_mask = obs_jax["timestep_pad_mask"]

        guide_input = None
        if not cfg.no_guide:
            with contextlib.suppress(KeyError):
                guide_input = lookup_guide(batch, cfg.guide_keys)

        bound = policy.model.module.bind({"params": policy.model.params})
        transformer_outputs = bound.crossformer_transformer(obs_jax, task_jax, pad_mask, train=False)

        policy.rng, key = jax.random.split(policy.rng)
        pred = bound.heads[policy.head_name].predict_action(
            transformer_outputs,
            rng=key,
            dof_ids=jnp.asarray(dof_ids),
            chunk_steps=jnp.asarray(chunk_steps)[None],
            train=False,
            guide_input=guide_input,
        )
        pred_actions = np.asarray(jax.device_get(pred))[0, -1]  # (H, A) — drop batch + window

        # remap to j0..j6 canonical order, then denormalize
        ids = dof_ids[0]
        pred_joints = slots_to_joints(pred_actions, ids)
        gt_joints = slots_to_joints(act[0], ids)
        pred_joints = np.stack(
            [denorm.denormalize_slot(pred_joints[t], np.array(JOINT_IDS), ds_name) for t in range(len(pred_joints))]
        )
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
        print(f"sample {i}: pred {pred_actions.shape} gt {gt_joints.shape}")

    if cfg.wandb.use:
        wandb.finish()
    print(Rule("done"))


if __name__ == "__main__":
    main(tyro.cli(Config))
