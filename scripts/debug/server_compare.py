"""Compare grain_full vs grain_raw+GrainlikeWrapper with model inference + rast video.

Loads a CrossFormerModel checkpoint, runs XFlowEvalLoop on both pipelines,
and logs rast videos to wandb for visual comparison.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path
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
from webpolicy.base_policy import BasePolicy
from webpolicy.server import Server

from crossformer import cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.cn.wab import Wandb
from crossformer.data.arec.arec import unpack_record
from crossformer.data.grain.loader import GrainDataFactory, make_source_by_mix
from crossformer.embody import MASK_ID
from crossformer.run.base_policy import action_dict_to_slots, ActionDenormWrapper, ModelPolicy
from crossformer.run.wrappers.grainlike import GrainlikeWrapper
from crossformer.run.xflow_eval import adapt_rast_batch
from crossformer.utils.callbacks.rast import RastConfig
from crossformer.utils.spec import ezdiff, ezvaldiff, spec
from crossformer.utils.tree import flat

from crossformer.run.rtc_policy import RTCPolicy

import wandb

log = logging.getLogger(__name__)


@dataclass
class Config(cn.Train):
    path: Path = tyro.MISSING
    step: int | None = None
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")
    eval_frames: int = 64
    wandb: Wandb = default(Wandb(project="crossformer-server-viz"))
    rast: RastConfig = default(RastConfig())
    use_guidance: bool = False
    host: str = "0.0.0.0"
    port: int | None = None

    flow_steps: int = 50
    head_name: str = "action"
    mse_batches: int = 8
    replay_skip: int = 0


class ReplayObsWrapper(BasePolicy):
    """Use the next raw dataset batch as model input, ignoring request observations."""

    def __init__(self, policy, ds_iter, *, skip: int = 0):
        self.policy = policy
        self.ds_iter = ds_iter
        self.skip = skip

    def reset(self, payload: dict | None = None) -> dict | None:
        inner = getattr(self.policy, "inner", None)
        if inner is None:
            return None
        return inner.reset(payload) if payload is not None else inner.reset()

    def warmup(self, *, accumulate: bool = False) -> dict:
        return self.step({}, accumulate=accumulate)

    def step(self, payload: dict, **kwargs) -> dict:
        print()
        print(spec(payload))
        raw = next(self.ds_iter)
        del raw["action"]
        del raw["info"]
        print(spec(raw))
        print()
        if "observation" in payload:
            ezvaldiff(raw, payload)
            raw["observation"]["proprio"] = payload["observation"]["proprio"]
            raw["observation"]["image"] = payload["observation"]["image"]
        for _ in range(self.skip):
            next(self.ds_iter)
        preprocessed = self.policy.preprocess_batch(raw)
        preprocessed = self.policy.shard_fn(jax.tree.map(jnp.array, preprocessed))
        if "task" in payload:
            preprocessed["task"] = payload["task"]
        result = self.policy.inner.step(preprocessed, **kwargs)
        return jax.tree.map(np.asarray, result)


def print_action_mse(label: str, policy, ds_iter, n: int) -> None:
    """Print normalized action MSE over n preprocessed batches."""
    sq_sum = 0.0
    count = 0

    for _ in range(n):
        batch = next(ds_iter)
        result = policy.step(batch)

        if isinstance(result["actions"], dict):
            result["actions"], _ = action_dict_to_slots(result["actions"], result["dof_ids"])

        pred = np.asarray(result["actions"], dtype=np.float32)
        gt = _get_gt_actions(batch, result)
        dof_ids = np.asarray(jax.device_get(result["dof_ids"]))

        if gt.ndim == 3:
            gt = gt[:, None, :, :]
        pred, gt = _align_action_tensors(pred, gt)

        err = (pred - gt) ** 2
        valid = np.broadcast_to((dof_ids != MASK_ID)[:, None, None, :], err.shape)
        sq_sum += float(err[valid].sum())
        count += int(valid.sum())

    mse_norm = sq_sum / count
    print(Rule(f"action MSE over {n} batches — {label}"))
    print({"mse_norm": round(float(mse_norm), 6)})


def _align_action_tensors(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop pred/gt to their shared chunk horizon."""
    steps = min(pred.shape[2], gt.shape[2])
    return pred[:, :, :steps, :], gt[:, :, :steps, :]


def _get_gt_actions(batch: dict, result: dict) -> np.ndarray:
    """Read GT actions from the batch or wrapper output."""
    act = batch.get("act")
    if act is None:
        act = result.get("act")
    if act is None:
        raise KeyError("Expected GT actions in batch['act'] or result['act']")
    return np.asarray(jax.device_get(act["base"]), dtype=np.float32)


def print_action_mse_by_dof_id(label: str, policy, ds_iter, n: int) -> None:
    """Print normalized action MSE grouped by dof_id over n preprocessed batches."""
    sq_sum = defaultdict(float)
    count = defaultdict(int)

    for _ in range(n):
        batch = next(ds_iter)
        result = policy.step(batch)
        if isinstance(result["actions"], dict):
            result["actions"], _ = action_dict_to_slots(result["actions"], result["dof_ids"])
        pred = np.asarray(result["actions"], dtype=np.float32)
        gt = _get_gt_actions(batch, result)
        dof_ids = np.asarray(jax.device_get(result["dof_ids"]))
        if gt.ndim == 3:
            gt = gt[:, None, :, :]
        pred, gt = _align_action_tensors(pred, gt)

        err = (pred - gt) ** 2
        for dof_id in sorted(np.unique(dof_ids[dof_ids != MASK_ID]).tolist()):
            mask = np.broadcast_to((dof_ids == dof_id)[:, None, None, :], err.shape)
            sq_sum[int(dof_id)] += float(err[mask].sum())
            count[int(dof_id)] += int(mask.sum())

            if dof_id == 8:
                print("gripper")
                print(pred[mask])
                print(gt[mask])

    out = {dof_id: round(float(sq_sum[dof_id] / count[dof_id]), 6) for dof_id in sorted(sq_sum)}

    print(Rule(f"action MSE by dof_id over {n} batches — {label}"))
    print(out)


def print_action_mse_by_chunk_step(label: str, policy, ds_iter, n: int) -> None:
    """Print normalized action MSE grouped by chunk step over n preprocessed batches."""
    sq_sum = defaultdict(float)
    count = defaultdict(int)

    for _ in range(n):
        batch = next(ds_iter)
        result = policy.step(batch)
        if isinstance(result["actions"], dict):
            result["actions"], _ = action_dict_to_slots(result["actions"], result["dof_ids"])
        pred = np.asarray(result["actions"], dtype=np.float32)
        gt = _get_gt_actions(batch, result)
        dof_ids = np.asarray(jax.device_get(result["dof_ids"]))
        if gt.ndim == 3:
            gt = gt[:, None, :, :]
        pred, gt = _align_action_tensors(pred, gt)

        err = (pred - gt) ** 2
        valid = np.broadcast_to((dof_ids != MASK_ID)[:, None, None, :], err.shape)
        for step in range(err.shape[2]):
            mask = valid[:, :, step : step + 1, :]
            sq_sum[step] += float(err[:, :, step : step + 1, :][mask].sum())
            count[step] += int(mask.sum())

    out = {step: round(float(sq_sum[step] / count[step]), 6) for step in sorted(sq_sum)}

    print(Rule(f"action MSE by chunk step over {n} batches — {label}"))
    print(out)


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
    print(spec(full_batch))

    # --- grain_raw + GrainlikeWrapper ---
    print(Rule("grain_raw + GrainlikeWrapper"))
    _, dconfig = make_source_by_mix(arec, cfg)
    proprio_keys = list(dconfig.keys.proprio.keys())
    max_a = max(arec.embodiment.action_dim for m in mix_items)

    print(Rule("building wrapper + policy"))

    """
    policy = ModelPolicy(
        str(cfg.path),
        step=cfg.step,
        head_name=cfg.head_name,
        guide_keys=cfg.guide_keys,
        use_guidance=cfg.use_guidance,
        flow_steps=cfg.flow_steps,
    )
    policy = ActionDenormWrapper(policy, loader_full.statistics[arec.name], embodiment=arec.embodiment)
    
    """
    model_policy = ModelPolicy(
            str(cfg.path),
            step=cfg.step,
            head_name=cfg.head_name,
            guide_keys=cfg.guide_keys,
            use_guidance=cfg.use_guidance,
            flow_steps=cfg.flow_steps,
    )
    policy = RTCPolicy(model_policy, d=1, s=3)
    policy = ActionDenormWrapper(policy, loader_full.statistics[arec.name], embodiment=arec.embodiment)



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
        norm_action=False,
    )

    def make_raw_ds(drop_actions: bool = False):
        ds, _ = make_source_by_mix(arec, cfg)
        return ds.to_iter_dataset(grain.ReadOptions(num_threads=4)).batch(
            cfg.data.loader.batch_size, drop_remainder=True
        )
        ds = grain.MapDataset.source(arec.source).seed(42)
        if isinstance(ds[0], bytes):
            ds = ds.map(unpack_record)

        # wrap flat arec format {image, proprio, info} into {observation, action, ...}
        # proprio is chunked to 20 future steps — proprio[1:] is the action, proprio[0] is observation
        def _wrap(x):
            proprio = x["proprio"]  # each value is (20, D)
            obs_proprio = {k: v[0] for k, v in proprio.items()}
            action = dict(proprio.items())
            out = {"observation": {"image": x["image"], "proprio": obs_proprio}, "info": x["info"], "action": action}
            return out

        sample = ds[0]
        if "observation" not in sample:
            ds = ds.map(_wrap)
        if drop_actions:
            ds = ds.map(lambda x: {k: v for k, v in x.items() if k != "action"})
        return ds.to_iter_dataset(grain.ReadOptions(num_threads=4)).batch(
            cfg.data.loader.batch_size, drop_remainder=True
        )



    policy.inner.start()


    # --- optional server ---
    if cfg.port is not None:
        # policy = ReplayObsWrapper(policy,
        # iter(make_source_by_mix(arec, cfg)[0].batch(1).to_iter_dataset(grain.ReadOptions(num_threads=4))) ,
        # skip=cfg.replay_skip)
        print(Rule("warmup"))
        policy.warmup()
        print(Rule(f"serving on {cfg.host}:{cfg.port}"))
        server = Server(policy, host=cfg.host, port=cfg.port)
        server.serve()

    policy.inner.stop()






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
    # print_action_mse("grain_full", policy.unwrapped(), iter(loader_full.dataset), cfg.mse_batches)
    # print_action_mse_by_dof_id("grain_full", policy.unwrapped(), iter(loader_full.dataset), cfg.mse_batches)
    # print_action_mse_by_chunk_step("grain_full", policy.unwrapped(), iter(loader_full.dataset), cfg.mse_batches)

    print_action_mse("grain_raw", policy, iter(make_raw_ds(drop_actions=False)), cfg.mse_batches)
    print_action_mse_by_dof_id("grain_raw", policy, iter(make_raw_ds(drop_actions=False)), cfg.mse_batches)
    print_action_mse_by_chunk_step("grain_raw", policy, iter(make_raw_ds(drop_actions=False)), cfg.mse_batches)

    quit()

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
