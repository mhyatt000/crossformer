"""Replay dataset samples through Policy.step and compare to dataset actions.

This is a throwaway debug script for checking train/server discrepancies without
starting the HTTP server. It loads dataset samples, feeds the latest
observation frame into `crossformer.run.server.Policy`, and compares the
returned current action against `act.base[:, 0, :]`.

Usage:
    uv run scripts/debug/server_replay_compare.py --path /path/to/checkpoint --task sweep
    uv run scripts/debug/server_replay_compare.py --path /path/to/checkpoint --task duck --mix xgym_duck_single
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import numpy as np
from rich import print
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.cn.dataset import DataSourceE
from crossformer.cn.dataset.dataset import Loader
from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory
from crossformer.run.server import Policy, PolicyConfig
from crossformer.run.train_step import lookup_guide
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer

np.set_printoptions(precision=2, suppress=True, linewidth=200)


@dataclass
class Config:
    path: str
    task: str
    step: int | None = None

    mix: str = "xgym_sweep_single"
    batch_size: int = 16
    n_batches: int = 4
    mp: int = 8

    chunk: int = 20
    exp: float = 0.99
    warmup: bool = False
    reset_history_per_sample: bool = True
    verbose_keys: bool = False
    skip_preprocess: bool = False
    mode: Literal["policy", "direct", "both"] = "policy"
    use_guidance: bool = False
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")


SKIP_DTYPES = {"O", "U", "S"}


def shard_batch(batch, mesh):
    """Shard a host-local batch across the data axis."""
    return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))


def _take_sample(x, idx: int, batch_size: int):
    """Take one sample from an array when it actually has a batch axis."""
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.shape[0] != batch_size:
        return arr
    return arr[idx]


def _copy_tree_arrays(x):
    if isinstance(x, dict):
        return {k: _copy_tree_arrays(v) for k, v in x.items()}
    return np.asarray(x)


def _format_like_example(value, example_value, idx: int, batch_size: int, skipped: set[str] | None, key: str):
    if isinstance(example_value, dict):
        src = value if isinstance(value, dict) else {}
        return {
            k: _format_like_example(src.get(k), sub_example, idx, batch_size, skipped, f"{key}/{k}")
            for k, sub_example in example_value.items()
        }

    if value is None:
        if skipped is not None:
            skipped.add(f"{key}:missing->example")
        return np.asarray(example_value)

    x = _take_sample(value, idx, batch_size)
    arr = np.asarray(x)
    if arr.dtype.kind in SKIP_DTYPES:
        if skipped is not None:
            skipped.add(f"{key}:{arr.dtype}->example")
        return np.asarray(example_value)

    target = np.asarray(example_value)
    if target.ndim >= 2 and arr.shape == target.shape[2:]:
        return arr[None, None]
    if target.ndim >= 1 and arr.shape == target.shape[1:]:
        return arr[None]
    return arr


def sample_obs(
    obs: dict,
    idx: int,
    batch_size: int,
    example_obs: dict,
    skipped: set[str] | None = None,
) -> dict:
    """Build one observation sample matching checkpoint example_batch shapes."""
    out = {}
    for key, example_value in example_obs.items():
        out[key] = _format_like_example(obs.get(key), example_value, idx, batch_size, skipped, key)
    return out


def sample_task(task: dict, idx: int, batch_size: int):
    """Slice one task entry while preserving a batch axis."""
    if not task:
        return {"pad_mask_dict": {}}

    def take(x):
        arr = np.asarray(x)
        if arr.ndim == 0 or arr.shape[0] != batch_size:
            return arr
        return arr[idx : idx + 1]

    return jax.tree.map(take, task)


def stats_dict(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def find_nonnumeric(tree: dict) -> list[str]:
    """Return non-numeric array leaves in a flat dict."""
    bad = []
    for key, value in tree.items():
        if isinstance(value, dict):
            nested = find_nonnumeric(value)
            bad.extend(f"{key}/{item}" for item in nested)
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in SKIP_DTYPES:
            bad.append(f"{key}:{arr.dtype}:{arr.shape}")
    return bad


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def add_stats_rows(table: Table, label: str, pred: np.ndarray, gt: np.ndarray) -> None:
    pred_stats = stats_dict(pred)
    gt_stats = stats_dict(gt)
    table.add_row(
        label,
        f"{mse(pred, gt):.6f}",
        f"{pred_stats['min']:.4f}",
        f"{pred_stats['max']:.4f}",
        f"{pred_stats['mean']:.4f}",
        f"{pred_stats['std']:.4f}",
        f"{gt_stats['min']:.4f}",
        f"{gt_stats['max']:.4f}",
        f"{gt_stats['mean']:.4f}",
        f"{gt_stats['std']:.4f}",
    )


def make_stats_table(title: str, pred_all, gt_all, pred_valid_all, gt_valid_all) -> Table:
    pred_all = np.stack(pred_all)
    gt_all = np.stack(gt_all)
    pred_valid = np.concatenate(pred_valid_all)
    gt_valid = np.concatenate(gt_valid_all)

    table = Table(title=title)
    table.add_column("slice", style="cyan")
    table.add_column("mse", justify="right")
    table.add_column("pred min", justify="right")
    table.add_column("pred max", justify="right")
    table.add_column("pred mean", justify="right")
    table.add_column("pred std", justify="right")
    table.add_column("gt min", justify="right")
    table.add_column("gt max", justify="right")
    table.add_column("gt mean", justify="right")
    table.add_column("gt std", justify="right")
    add_stats_rows(table, "all dims", pred_all.reshape(-1), gt_all.reshape(-1))
    add_stats_rows(table, "valid dims", pred_valid, gt_valid)
    return table


def sample_guide(batch: dict, idx: int, batch_size: int, guide_keys: tuple[str, ...]) -> np.ndarray | None:
    try:
        guide = np.asarray(lookup_guide(batch, guide_keys))
    except Exception:
        return None
    if guide.ndim == 0 or guide.shape[0] != batch_size:
        return None
    return guide[idx : idx + 1]


def predict_direct(
    policy: Policy,
    obs: dict,
    task: dict,
    dof_ids: np.ndarray,
    chunk_steps: np.ndarray,
    guide_input: np.ndarray | None,
) -> np.ndarray:
    """Run the model via the already-jitted sample_actions path."""
    obs = jax.tree.map(lambda x: jnp.asarray(x), obs)
    task = jax.tree.map(lambda x: jnp.asarray(x), task)
    policy.rng, key = jax.random.split(policy.rng)
    pred = policy.model.sample_actions(
        observations=obs,
        tasks=task,
        head_name=policy.head_name,
        rng=key,
        dof_ids=jnp.asarray(dof_ids)[None],
        chunk_steps=jnp.asarray(chunk_steps)[None],
        train=False,
    )
    return np.asarray(pred[0, -1, 0], dtype=np.float32)


def main(cfg: Config):
    mesh = Mesh(jax.devices(), axis_names="batch")

    _apply_fd_limit(512**2)
    train_cfg = cn.Train(
        data=cn.Dataset(
            mix=DataSourceE[cfg.mix],
            loader=Loader(use_grain=True, global_batch_size=cfg.batch_size),
        ),
        seed=42,
        verbosity=0,
    )
    dataset = GrainDataFactory(mp=cfg.mp).make(train_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=True)
    dsit = iter(dataset.dataset)

    policy = Policy(
        PolicyConfig(
            path=cfg.path,
            task=cfg.task,
            step=cfg.step,
            chunk=cfg.chunk,
            exp=cfg.exp,
            warmup=cfg.warmup,
        )
    )
    if cfg.skip_preprocess:
        policy.preprocess = lambda obs: obs

    # normalized (raw model output vs normalized gt)
    norm_pred_all, norm_gt_all = [], []
    norm_pred_valid_all, norm_gt_valid_all = [], []
    # denormalized
    raw_pred_all, raw_gt_all = [], []
    raw_pred_valid_all, raw_gt_valid_all = [], []
    srv_pred_all, srv_gt_all = [], []
    srv_pred_valid_all, srv_gt_valid_all = [], []
    skipped_obs_keys: set[str] = set()
    example_obs = jax.device_get(policy.model.example_batch["observation"])
    denorm = ActionBatchDenormalizer(policy.model.dataset_statistics)

    # print denorm stats for sanity check
    ds_stats = denorm._action_stats(policy.dataset_name)
    for part_name, part_stat in ds_stats.items():
        if not isinstance(part_stat, dict):
            s = part_stat if hasattr(part_stat, "mean") else type(part_stat)
            print(
                f"  denorm stats [{policy.dataset_name}][{part_name}]: mean={np.asarray(getattr(s, 'mean', None))}, std={np.asarray(getattr(s, 'std', None))}"
            )

    print(Rule("server replay compare"))
    print(
        {
            "path": cfg.path,
            "task": cfg.task,
            "mix": cfg.mix,
            "batch_size": cfg.batch_size,
            "n_batches": cfg.n_batches,
            "policy_chunk": cfg.chunk,
            "reset_history_per_sample": cfg.reset_history_per_sample,
            "skip_preprocess": cfg.skip_preprocess,
            "mode": cfg.mode,
            "use_guidance": cfg.use_guidance,
        }
    )

    for batch_idx in tqdm(range(cfg.n_batches)):
        batch = jax.device_get(next(dsit))
        act = np.asarray(batch["act"]["base"], dtype=np.float32)
        dof_ids = np.asarray(batch["act"]["id"])
        task = batch.get("task", {"pad_mask_dict": {}})
        B, H, _ = act.shape
        chunk_steps = np.arange(H, dtype=np.float32)

        for sample_idx in tqdm(range(B), leave=False):
            if cfg.reset_history_per_sample:
                policy.reset_history()

                policy.task = sample_task(task, sample_idx, B)
            payload = {
                "observation": sample_obs(batch["observation"], sample_idx, B, example_obs, skipped_obs_keys),
                "dof_ids": dof_ids[sample_idx],
                "chunk_steps": chunk_steps,
            }
            bad_obs = find_nonnumeric(payload["observation"])
            if bad_obs:
                print({"sample_idx": sample_idx, "bad_observation_keys": bad_obs})
            if cfg.mode in {"direct", "both"}:
                guide_input = sample_guide(batch, sample_idx, B, cfg.guide_keys) if cfg.use_guidance else None
                norm_pred = predict_direct(
                    policy,
                    payload["observation"],
                    policy.task,
                    dof_ids[sample_idx],
                    chunk_steps,
                    guide_input,
                )
                norm_gt = act[sample_idx, 0]
                valid = dof_ids[sample_idx] != 0
                # normalized
                norm_pred_all.append(norm_pred)
                norm_gt_all.append(norm_gt)
                norm_pred_valid_all.append(norm_pred[valid])
                norm_gt_valid_all.append(norm_gt[valid])
                # denormalized
                denorm_pred = denorm.denormalize_slot(norm_pred, dof_ids[sample_idx], policy.dataset_name)
                denorm_gt = denorm.denormalize_slot(norm_gt, dof_ids[sample_idx], policy.dataset_name)
                raw_pred_all.append(denorm_pred)
                raw_gt_all.append(denorm_gt)
                raw_pred_valid_all.append(denorm_pred[valid])
                raw_gt_valid_all.append(denorm_gt[valid])

            if cfg.mode in {"policy", "both"}:
                srv_pred = np.asarray(policy.step(payload)["actions"], dtype=np.float32)
                valid = dof_ids[sample_idx] != 0
                # denormalized
                srv_denorm_gt = denorm.denormalize_slot(act[sample_idx, 0], dof_ids[sample_idx], policy.dataset_name)
                srv_pred_all.append(srv_pred)
                srv_gt_all.append(srv_denorm_gt)
                srv_pred_valid_all.append(srv_pred[valid])
                srv_gt_valid_all.append(srv_denorm_gt[valid])

        print(f"batch {batch_idx}: compared {B} samples")

    if cfg.mode in {"direct", "both"}:
        print(make_stats_table("Direct NORMALIZED", norm_pred_all, norm_gt_all, norm_pred_valid_all, norm_gt_valid_all))
        print(make_stats_table("Direct DENORMALIZED", raw_pred_all, raw_gt_all, raw_pred_valid_all, raw_gt_valid_all))
    if cfg.mode in {"policy", "both"}:
        print(
            make_stats_table(
                "Server policy vs act.base[:, 0, :]",
                srv_pred_all,
                srv_gt_all,
                srv_pred_valid_all,
                srv_gt_valid_all,
            )
        )
    if skipped_obs_keys or cfg.verbose_keys:
        print({"skipped_obs_keys": sorted(skipped_obs_keys)})


if __name__ == "__main__":
    main(tyro.cli(Config))
