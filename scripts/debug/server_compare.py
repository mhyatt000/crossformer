"""Compare grain_full vs grain_raw+GrainlikeWrapper with model inference + rast video.

Loads a CrossFormerModel checkpoint, runs XFlowEvalLoop on both pipelines,
and logs rast videos to wandb for visual comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Literal

import grain
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import numpy as np
from rich import print
from rich.rule import Rule
from tqdm import tqdm
import tyro
from webpolicy.server import Server

from crossformer import cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.cn.wab import Wandb
from crossformer.data.arec.arec import unpack_record
from crossformer.data.grain.loader import GrainDataFactory, make_source_by_mix
from crossformer.embody import MASK_ID
from crossformer.run.base_policy import ActionDenormWrapper, ModelPolicy
from crossformer.run.wrappers.grainlike import GrainlikeWrapper
from crossformer.run.xflow_eval import adapt_rast_batch
from crossformer.utils.callbacks.rast import RastConfig
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer
from crossformer.utils.spec import ezdiff, ezvaldiff
from crossformer.utils.tree import flat
import wandb

log = logging.getLogger(__name__)


@dataclass
class Config(cn.Train):
    path: Path = tyro.MISSING
    step: int | None = None
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    obs_keys: tuple[str, ...] = ("proprio_.*",)
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")
    eval_frames: int = 64
    wandb: Wandb = default(Wandb(project="crossformer-server-viz"))
    rast: RastConfig = default(RastConfig())
    use_guidance: bool = False
    host: str = "0.0.0.0"
    port: int | None = None


def resolve_obs_keys(obs, patterns):
    keys = []
    for pat in patterns:
        keys.extend(k for k in sorted(obs) if k not in keys and re.fullmatch(pat, k))
    return tuple(keys)


def print_action_mse(label: str, policy, batch, stats) -> None:
    """Print normalized and denormalized action MSE. batch must be preprocessed."""
    result = policy.step(batch)
    pred_np = np.asarray(result["actions"], dtype=np.float32)[:, 0, 0]  # (B, A) normalized
    gt = np.asarray(jax.device_get(batch["act"]["base"]), dtype=np.float32)
    if gt.ndim == 3:
        gt = gt[:, None, :, :]  # (B, H, A) -> (B, 1, H, A)
    gt_np = gt[:, 0, 0]  # (B, A)
    dof_ids_np = np.asarray(jax.device_get(result["dof_ids"]))

    mse_norm = float(np.mean((pred_np - gt_np) ** 2))

    denorm = ActionBatchDenormalizer(stats)
    ds_names = denorm.decode_dataset_names(jax.device_get(batch["info"]["dataset_name"]))

    pred_d, gt_d = [], []
    for i, ds_name in enumerate(ds_names):
        valid = dof_ids_np[i] != MASK_ID
        pd = denorm.denormalize_slot(pred_np[i], dof_ids_np[i], ds_name)
        gd = denorm.denormalize_slot(gt_np[i], dof_ids_np[i], ds_name)
        pred_d.append(pd[valid])
        gt_d.append(gd[valid])

    mse_denorm = float(np.mean((np.concatenate(pred_d) - np.concatenate(gt_d)) ** 2))
    print(Rule(f"action MSE — {label}"))
    print({"mse_norm": round(mse_norm, 6), "mse_denorm": round(mse_denorm, 6)})


def run_rast(label: str, ds_iter, cfg: Config, policy) -> None:
    """Render rast videos. policy must be an ActionDenormWrapper — flow is already denormed."""
    rast_cb = cfg.rast.create()
    per_cam: list[list[np.ndarray]] | None = None
    frames_left = cfg.eval_frames

    bar = tqdm(total=cfg.eval_frames, desc=f"Rendering {label} rast videos")
    while frames_left > 0:
        raw = next(ds_iter)
        result = policy.step(raw, accumulate=True)
        if "act" in raw:
            act = raw["act"]
        elif "act" in result:
            act = result["act"]
        else:
            ids = np.asarray(result["dof_ids"])  # (B, max_a)
            act = {"base": np.zeros((*ids.shape[:-1], 1, 1, ids.shape[-1]), dtype=np.float32), "id": ids}
        pred_flow = result["actions"]
        rast_batch, keep = adapt_rast_batch(act, pred_flow)
        if rast_batch is None:
            continue

        for local_idx in range(min(frames_left, len(keep))):
            chunk = rast_batch["predict"][-1, local_idx]
            print(chunk)
            if chunk.ndim == 3:
                chunk = chunk[0]
            traj_frames = rast_cb.render_trajectory(chunk)
            if per_cam is None:
                per_cam = [[] for _ in range(len(traj_frames))]
            for ci, frame in enumerate(traj_frames):
                per_cam[ci].append(frame)
            frames_left -= 1
            bar.update(1)

    if per_cam is None:
        return
    for ci, frames in enumerate(per_cam):
        video = np.stack(frames, axis=0)
        cfg.wandb.log({f"{label}/rast/cam_{ci}": wandb.Video(np.moveaxis(video, -1, 1), fps=10)})


def main(cfg: Config) -> None:
    mix_items = cfg.data.mix.value.flatten()
    arec = Arec.from_name(mix_items[0][0])

    # sharding setup — replicate batch across GPUs like xflow.py
    mesh = Mesh(jax.devices(), axis_names="batch")

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    run = cfg.wandb.initialize(cfg, name="server-compare")

    # --- grain_full dataset ---
    print(Rule("grain_full dataset"))
    factory = GrainDataFactory(
        mp=0,
        shuffle=False,
        mask_slot=False,
        shuffle_slot=False,
        imaug=False,
    )
    loader_full = factory.make(cfg, shard_fn=shard, train=False)
    full_batch = next(iter(loader_full.dataset))
    obs_keys = resolve_obs_keys(full_batch["observation"], cfg.obs_keys)
    print(f"  obs_keys: {obs_keys}")

    # --- grain_raw + GrainlikeWrapper ---
    print(Rule("grain_raw + GrainlikeWrapper"))
    _, dconfig, _ = make_source_by_mix(arec, cfg)
    proprio_keys = list(dconfig.keys.proprio.keys())
    max_a = max(Arec.from_name(m[0]).embodiment.action_dim for m in mix_items)

    print(Rule("building wrapper + policy"))
    policy = ModelPolicy(str(cfg.path), step=cfg.step, guide_keys=cfg.guide_keys, use_guidance=cfg.use_guidance)
    policy = ActionDenormWrapper(policy, loader_full.dataset_statistics, dataset_name=arec.name)
    policy = GrainlikeWrapper(
        policy,
        dataset_name=arec.name,
        embodiment=arec.embodiment,
        max_a=max_a,
        stats=loader_full.statistics[arec.name],
        proprio_keys=proprio_keys,
        skip_norm_keys=cfg.data.transform.skip_norm_keys,
        resize_to=64,
        shard_fn=None if cfg.port is not None else shard,
    )

    def make_raw_ds(drop_actions: bool = False):
        ds = grain.MapDataset.source(arec.source).seed(42).map(unpack_record)
        if drop_actions:
            ds = ds.map(lambda x: {k: v for k, v in x.items() if k != "action"})
        return ds.to_iter_dataset(grain.ReadOptions(num_threads=4)).batch(
            cfg.data.loader.batch_size, drop_remainder=True
        )

    # --- optional server ---
    if cfg.port is not None:
        print(Rule(f"serving on {cfg.host}:{cfg.port}"))
        server = Server(policy, host=cfg.host, port=cfg.port)
        server.serve()

    # --- diff ---
    print(Rule("comparing first batch"))
    raw_batch = next(iter(make_raw_ds(drop_actions=False)))
    raw_preprocessed = policy.preprocess_batch(raw_batch)
    full_s = {k: v[0] for k, v in flat(full_batch).items()}
    raw_s = {k: v[0] for k, v in flat(raw_preprocessed).items()}
    print("\n[bold]ezdiff (spec):[/bold]")
    ezdiff(full_s, raw_s, simple=True)
    print("\n[bold]ezvaldiff (values):[/bold]")
    ezvaldiff(full_s, raw_s)

    # --- action MSE ---
    raw_sharded = shard(jax.tree.map(jnp.array, raw_batch))
    raw_preprocessed_sharded = policy.preprocess_batch(raw_sharded)
    print_action_mse("grain_full", policy.unwrapped(), full_batch, loader_full.dataset_statistics)
    print_action_mse("grain_raw", policy.unwrapped(), raw_preprocessed_sharded, loader_full.dataset_statistics)

    # --- warmup JIT for both modes ---
    # print(Rule("JIT warmup"))
    # policy.unwrapped().warmup(accumulate=False)
    # policy.unwrapped().warmup(accumulate=True)

    # --- rast videos ---
    print(Rule("rast: grain_raw"))
    run_rast("grain_raw", iter(make_raw_ds(drop_actions=True)), cfg, policy)

    print(Rule("rast: grain_full"))
    # this uses all except for GrainlikeWrapper
    run_rast("grain_full", iter(loader_full.dataset), cfg, policy.inner)

    print(Rule("done"))
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
