from __future__ import annotations

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
"""Debug script: run N training steps then render flow PCA GIF via VizCallback.

Usage:
    uv run scripts/debug/viz_flow.py --n_steps 10
    uv run scripts/debug/viz_flow.py --n_steps 0  # skip training, forward-pass only
    uv run scripts/debug/viz_flow.py --out_dir /tmp/mytest

Mirrors the setup and eval logic from scripts/finetune.py without wandb or checkpointing.
"""

from dataclasses import dataclass
from functools import partial
import logging
from pathlib import Path
from typing import Any

from einops import rearrange
import flax
import imageio.v2 as imageio
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import ConfigDict
import optax
import tensorflow as tf
import tyro

from crossformer import cn
from crossformer.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.callbacks.viz import VizCallback
from crossformer.utils.deco import deprecate
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    merge_params,
    process_text,
    TrainState,
)

log = logging.getLogger(__name__)


@dataclass
class Config(cn.Train):
    n_steps: int = 10  # training steps before viz (0 = forward-pass only)
    out_dir: Path = Path("/tmp/crossformer")


# ---------------------------------------------------------------------------
# Setup helpers (mirrors finetune.py verbatim)
# ---------------------------------------------------------------------------


def _build_mesh():
    mesh = Mesh(jax.devices(), axis_names="batch")
    dp = NamedSharding(mesh, PartitionSpec("batch"))
    rep = NamedSharding(mesh, PartitionSpec())
    return mesh, dp, rep


def _load_model(cfg: Config, config_out: dict) -> tuple[CrossFormerModel, dict]:
    pretrained = CrossFormerModel.load_pretrained(cfg.pretrained_path, step=cfg.pretrained_step)
    flat = flax.traverse_util.flatten_dict(pretrained.config, keep_empty_nodes=True)
    flat = cfg.model.delete(flat)
    config = ConfigDict(flax.traverse_util.unflatten_dict(flat))
    config.update(cfg.model.create())
    config = config.to_dict()
    check_config_diff(config, pretrained.config)
    return pretrained, config


def _build_dataset(cfg: Config, mesh: Mesh, dp_sharding: NamedSharding):
    @deprecate("let grain do it", strict=False)
    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    if cfg.data.loader.use_grain:
        from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory

        dataset = GrainDataFactory().make(cfg, shard_fn=shard, train=True)
        _apply_fd_limit(512**2)
        return dataset, iter(dataset.dataset), shard
    else:
        dataset = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS, train=True)

        text_processor_cfg = {"text_processor": None}  # placeholder; overridden below
        text_processor = None

        def process_batch(batch):
            batch = process_text(batch, text_processor)
            del batch["dataset_name"]
            return batch

        dsit = map(shard, map(process_batch, dataset.iterator(prefetch=cfg.data.loader.prefetch)))
        return dataset, dsit, shard


# ---------------------------------------------------------------------------
# Loss + train_step — exact copy of finetune.py for fidelity
# ---------------------------------------------------------------------------


def _make_train_step(model: CrossFormerModel, rep_sharding, dp_sharding):
    def loss_fn(params, batch, rng, train=True):
        bound = model.module.bind({"params": params}, rngs={"dropout": rng})
        embeddings = bound.crossformer_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        # quick patch (mirrors finetune.py)
        batch["action"] = jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), batch["action"])
        if "k3ds" in batch["action"]:
            batch["action"]["k3ds"] = rearrange(batch["action"]["k3ds"], "b w h x y -> b w h (x y)")
        batch["embodiment"] = jax.tree.map(lambda x: jnp.squeeze(x, axis=-1), batch["embodiment"])

        total_loss, metrics = jnp.zeros(()), {}
        matched = 0
        for head_name, head in bound.heads.items():
            if head_name not in batch["action"] or head_name not in batch["embodiment"]:
                continue
            if head_name == "single_arm":
                head_loss, head_metrics = head.loss(embeddings=embeddings, batch=batch, train=True)
            else:
                head_loss, head_metrics = head.loss(
                    embeddings,
                    batch["action"][head_name],
                    batch["observation"]["timestep_pad_mask"],
                    action_pad_mask=jnp.ones_like(batch["action"][head_name], dtype=jnp.bool_),
                    action_head_mask=batch["embodiment"][head_name],
                    train=train,
                )
                frac = batch["embodiment"][head_name].mean()
                total_loss = total_loss + head_loss * frac * head.loss_weight
            metrics[head_name] = head_metrics
            matched += 1

        if matched == 0:
            raise ValueError(f"No overlapping heads: model={tuple(bound.heads)}, batch={tuple(batch['action'])}")
        metrics["total_loss"] = total_loss
        return total_loss, metrics

    @partial(jax.jit, in_shardings=[rep_sharding, dp_sharding])
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng, train=True)
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        info["grad_norm"] = grad_norm
        info["update_norm"] = optax.global_norm(updates)
        return state.apply_gradients(grads=grads, rng=rng), info

    return train_step


# ---------------------------------------------------------------------------
# Viz eval — exact copy of finetune.py eval block
# ---------------------------------------------------------------------------


def _run_viz(
    model: CrossFormerModel,
    dsit,
    train_state: TrainState,
    viz_cb: VizCallback,
    out_dir: Path,
    step: int,
) -> Path | None:
    try:
        val_batch = next(dsit)
        viz_rng = jax.random.fold_in(train_state.rng, step)
        bound = model.module.bind({"params": train_state.model.params}, rngs={"dropout": viz_rng})
        embeddings = bound.crossformer_transformer(
            val_batch["observation"],
            val_batch["task"],
            val_batch["observation"]["timestep_pad_mask"],
            train=False,
        )

        flow_head_name = next(
            (name for name, head in bound.heads.items() if hasattr(head, "flow_steps") and name in val_batch["action"]),
            None,
        )
        if flow_head_name is None:
            flow_head_name = next((name for name in bound.heads if name in val_batch["action"]), None)
        if flow_head_name is None:
            raise ValueError(f"No usable head. model={tuple(bound.heads)}, batch={tuple(val_batch['action'])}")

        pred_flow = bound.heads[flow_head_name].predict_action(embeddings, train=False, rng=viz_rng, sample_shape=(1,))
        pred_flow = jax.device_get(pred_flow)
        if pred_flow.shape[0] == 1:
            pred_flow = pred_flow.squeeze(0)
        if pred_flow.ndim not in (5, 6):
            raise ValueError(f"Expected flow ndim 5 or 6, got {pred_flow.shape} from head {flow_head_name}")

        gt_action = val_batch["action"][flow_head_name]
        viz_batch = {"pred_flow": pred_flow, "gt_action": gt_action}

        frames = viz_cb(viz_batch)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"flow_pca_step_{step}.gif"
        imageio.mimsave(str(out_path), frames, fps=10, loop=0)
        print(f"[viz_flow] Saved GIF → {out_path}")
        return out_path

    except Exception:
        log.exception("VizCallback failed at step %d", step)
        return None


def _extract_joints(val_batch: dict, pred_flow: Any) -> Any:
    """Mirror finetune.py joint extraction logic."""

    def pick_joints(x):
        if x is None:
            return None
        d = x.shape[-1]
        if d >= 7:
            return x[..., :7]
        return None

    obs = val_batch.get("observation", {})
    prop = obs.get("proprio")
    joints = None

    if isinstance(prop, dict):
        for key in ("joints", "single", "single_arm"):
            if key in prop:
                joints = pick_joints(prop[key])
                break
    elif prop is not None:
        joints = pick_joints(prop)
    elif "proprio_single" in obs:
        joints = pick_joints(obs["proprio_single"])
    elif "proprio_single_arm" in obs:
        joints = pick_joints(obs["proprio_single_arm"])

    if joints is not None and joints.shape[-1] >= 7:
        joints = jax.device_get(joints)
        if joints.ndim == 2:
            joints = joints[:, None, None, :]
        elif joints.ndim == 3:
            joints = joints[:, :, None, :]
        return joints

    # Fallback: use the flow itself as proxy joints
    print("[viz_flow] WARNING: no usable proprio joints found; using flow[0] as base joints fallback.")
    return pred_flow[:, 0] if pred_flow.ndim == 6 else pred_flow[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config) -> None:
    tf.random.set_seed(cfg.seed + jax.process_index())
    tf.config.set_visible_devices([], "GPU")
    initialize_compilation_cache()

    mesh, dp_sharding, rep_sharding = _build_mesh()

    # Model
    pretrained_model, config = _load_model(cfg, {})
    text_processor = None if config["text_processor"] is None else ModuleSpec.instantiate(config["text_processor"])()

    # Data
    dataset, dsit, _ = _build_dataset(cfg, mesh, dp_sharding)

    example_batch = next(dsit)
    print("[viz_flow] batch spec:", spec(example_batch))

    # Init model
    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    model = CrossFormerModel.from_config(
        config,
        example_batch,
        text_processor=text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged)
    del pretrained_model

    # Optimizer
    if cfg.optimizer.frozen_keys is None:
        cfg.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]
    tx, _, _ = create_optimizer(model.params, **cfg.optimizer.create())
    train_state = TrainState.create(model=model, tx=tx, rng=rng)

    train_step = _make_train_step(model, rep_sharding, dp_sharding)

    viz_cb = VizCallback(flow_key=("pred_flow",), base_key=("gt_action",))

    # Training loop
    for i in range(cfg.n_steps):
        batch = next(dsit)
        train_state, info = train_step(train_state, batch)
        loss = jax.device_get(info.get("total_loss", float("nan")))
        print(f"[viz_flow] step {i + 1}/{cfg.n_steps}  loss={loss:.4f}")

    # Viz
    print(f"[viz_flow] running VizCallback after {cfg.n_steps} steps …")
    _run_viz(model, dsit, train_state, viz_cb, cfg.out_dir, step=cfg.n_steps)


if __name__ == "__main__":
    main(tyro.cli(Config))
