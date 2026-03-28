"""End-to-end XFlowHead training: CrossFormerModel + real data.

Bundled action format: uses act.base / act.id from the grain embody
pipeline instead of per-head action extraction.

Usage:
    uv run scripts/train/xflow.py
    uv run scripts/train/xflow.py --steps 500 --lr 3e-4
    uv run scripts/train/xflow.py --mix xgym_sweep --batch-size 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import re

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
import optax
from rich import print
from rich.rule import Rule
from rich.table import Table
import tensorflow as tf
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.data.grain.embody import decode_embody_name
from crossformer.model.components.guidance import TokenGuidance
from crossformer.model.components.heads.dof import build_query_mask
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.tokenizers import LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.train_step import make_train_step
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.train_utils import Timer, TrainState
from crossformer.utils.tree.core import flat
import wandb

from crossformer.model.components.heads.xflow_rtc import rtc_predict_action

# -- config -------------------------------------------------------------------


@dataclass
class Config:
    """XFlowHead end-to-end training config."""

    name: str = ""
    steps: int = 1  # training steps (1 for debug)
    lr: float = 1e-3  # learning rate
    log_every: int = 1  # log interval
    batch_size: int = 4  # global batch size
    mix: str = "xgym_sweep"  # dataset mix name
    horizon: int = 20  # action horizon from data pipeline
    transformer_size: str = "dummy"  # transformer size preset
    obs_keys: tuple[str, ...] = ("proprio_.*", "time", "timestep")  # lowdim obs keys to tokenize

    # Token guidance
    use_guidance: bool = True  # enable guidance tokens
    compress_guidance: bool = False  # compress via perceiver latents
    num_guidance_latents: int = 4  # latent count when compress=True
    guide_key: str = "action.pose"  # dot-path into batch for guidance signal

    mp: int = 1  # grain multiproc (for data loading)
    wandb: cn.Wandb = field(default_factory=cn.Wandb)


# -- helpers ------------------------------------------------------------------


def make_model_config(cfg, max_h, max_a, max_w):
    """Build CrossFormerModel config with XFlowHead."""
    token_dim, transformer_kwargs = common_transformer_sizes(cfg.transformer_size)
    readout_name = "xflow"
    readout_key = f"readout_{readout_name}"
    return {
        "model": {
            "observation_tokenizers": {
                "lowdim": ModuleSpec.create(
                    LowdimObsTokenizer,
                    obs_keys=list(cfg.obs_keys),
                ),
            },
            "task_tokenizers": {},
            "heads": {
                "xflow": ModuleSpec.create(
                    XFlowHead,
                    readout_key=readout_key,
                    max_dofs=max_a,
                    max_horizon=max_h,
                    num_query_channels=256,
                    num_heads=8,
                    num_self_attend_layers=2,
                    flow_steps=10,
                ),
            },
            "readouts": {readout_name: 4},
            "token_embedding_size": token_dim,
            "transformer_kwargs": transformer_kwargs,
            "max_horizon": max_w,
        },
        "text_processor": None,
    }


def normalize_obs(obs, obs_keys):
    """Flatten selected lowdim inputs to (B, W, D)."""
    out = dict(obs)
    for key in obs_keys:
        x = out[key]
        if x.ndim == 2:
            out[key] = x[..., None]
        elif x.ndim > 3:
            out[key] = x.reshape(*x.shape[:2], -1)
    return out


def resolve_obs_keys(obs, patterns):
    """Resolve regex patterns against real observation keys."""
    keys = []
    for pat in patterns:
        keys.extend(k for k in sorted(obs) if k not in keys and re.fullmatch(pat, k))
    if not keys:
        raise ValueError(f"No observation keys matched {patterns}. available={tuple(sorted(obs))}")
    return tuple(keys)


def extract_bundled_actions(batch, max_h):
    """Extract bundled actions from grain embody pipeline.

    Returns:
        actions: (B, W, H, max_a) — W=1 added if missing.
        dof_ids: (B, max_a) — from act.id.
        chunk_steps: (B, H) — just arange(H) for now.
            NOTE: chunk_steps padding disabled — all positions are valid.
    """
    actions = batch["act"]["base"]  # (B, H, max_a)
    if actions.ndim == 3:
        actions = actions[:, None, :, :]  # (B, 1, H, max_a)
    B = actions.shape[0]
    H = actions.shape[2]

    dof_ids = batch["act"]["id"]  # (B, max_a)
    # NOTE: no chunk_steps padding — just dense arange(H)
    chunk_steps = jnp.tile(jnp.arange(H, dtype=jnp.float32)[None], (B, 1))

    return actions, dof_ids, chunk_steps


def per_embodiment_metrics(batch, update_info):
    """Compute per-embodiment loss from sample_mse and act.embody.

    Returns dict like {"embodiment/single": mse, "embodiment/dual_arm": mse, ...}.
    """
    embody_arr = np.array(batch["act"]["embody"])  # (B, 32) uint8
    sample_mse = np.array(update_info["sample_mse"])  # (B,)
    names = [decode_embody_name(embody_arr[i]) for i in range(embody_arr.shape[0])]
    groups: dict[str, list[float]] = {}
    for name, mse in zip(names, sample_mse):
        groups.setdefault(name, []).append(float(mse))
    return {f"embodiment/{k}": sum(v) / len(v) for k, v in groups.items()}


def shard_batch(batch, mesh):
    """Shard a host-local batch across the data axis."""
    return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))


# -- main ---------------------------------------------------------------------


def main(cfg: Config):
    tf.config.set_visible_devices([], "GPU")
    initialize_compilation_cache()
    devices = jax.devices()
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    print(Rule("XFlowHead + CrossFormerModel: bundled actions", style="bold magenta"))
    print(f"  backend={jax.default_backend()} devices={len(devices)}")
    if cfg.batch_size % len(devices) != 0:
        raise ValueError(f"batch_size={cfg.batch_size} must be divisible by devices={len(devices)}")

    max_h = cfg.horizon
    run = cfg.wandb.initialize(cfg)

    # Load data
    print(Rule("loading data"))
    from crossformer.cn.dataset import DataSourceE
    from crossformer.cn.dataset.dataset import Loader
    from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory

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
    example_batch = next(dsit)
    print(spec(example_batch))
    obs_keys = resolve_obs_keys(example_batch["observation"], cfg.obs_keys)
    print(f"  obs_keys: {obs_keys}")

    # Get max_a from the bundled action shape
    max_a = example_batch["act"]["id"].shape[-1]
    print(f"  max_h={max_h}  max_a={max_a}")
    print(f"  act.base shape: {example_batch['act']['base'].shape}")
    print(f"  act.id   shape: {example_batch['act']['id'].shape}")

    # Build model
    print(Rule("building CrossFormerModel"))
    max_w = example_batch["observation"]["timestep_pad_mask"].shape[1]
    example_obs = normalize_obs(example_batch["observation"], obs_keys)
    config = make_model_config(cfg, max_h, max_a, max_w)
    config["model"]["observation_tokenizers"]["lowdim"] = ModuleSpec.create(
        LowdimObsTokenizer,
        obs_keys=[f"^{re.escape(k)}$" for k in obs_keys],
    )
    init_batch = {
        "observation": example_obs,
        "task": example_batch.get("task", {"pad_mask_dict": {}}),
    }
    wandb.config.update(
        {
            "example_batch_spec": spec(example_batch),
            "obs_keys": obs_keys,
            "max_a": max_a,
            "model_config": config,
        },
        allow_val_change=True,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, train_rng, pred_rng = jax.random.split(rng, 3)

    model = CrossFormerModel.from_config(
        config,
        init_batch,
        text_processor=None,
        verbose=False,
        rng=init_rng,
    )
    model = model.replace(
        params=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.params),
        example_batch=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.example_batch),
    )
    n_params = sum(x.size for x in jax.tree.leaves(model.params))
    print(f"  params: {n_params:,}")
    print(f"  heads: {list(model.module.heads.keys())}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # Guidance encoder
    guide_module = None
    guide_params = None
    if cfg.use_guidance:
        print(Rule("guidance encoder"))
        guide_example = flat(example_batch).get(cfg.guide_key)
        if guide_example is None:
            raise ValueError(f"guide_key={cfg.guide_key!r} not found in batch. check dot-path.")
        if guide_example.ndim > 3:
            guide_example = guide_example.reshape(*guide_example.shape[:2], -1)
        print(f"  guide_key={cfg.guide_key} shape={guide_example.shape}")
        guide_module = TokenGuidance(
            embed_dim=256,
            compress=cfg.compress_guidance,
            num_latents=cfg.num_guidance_latents,
        )
        guide_vars = guide_module.init(init_rng, guide_example)
        guide_params = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), guide_vars["params"])
        n_guide = sum(x.size for x in jax.tree.leaves(guide_params))
        print(f"  guide params: {n_guide:,}  compress={cfg.compress_guidance}")
        wandb.config.update({"n_guide_params": n_guide}, allow_val_change=True)

    # Optimizer + state
    combined_params = {"model": model.params, "guide": guide_params} if guide_module is not None else model.params
    tx = optax.adamw(cfg.lr, weight_decay=1e-4)
    state = TrainState.create(model=model.replace(params=combined_params), tx=tx, rng=train_rng)
    train_step = make_train_step(model.module, cfg.lr, guide_module=guide_module)

    # Train
    print(Rule("training"))
    table = Table(title="training")
    table.add_column("step", justify="right", style="cyan")
    table.add_column("loss", justify="right")
    table.add_column("|grad|", justify="right", style="dim")

    losses = []
    timer = Timer()
    for step in tqdm(range(cfg.steps)):
        timer.tick("total")
        with timer("dataset"):
            batch = next(dsit)
            obs = normalize_obs(batch["observation"], obs_keys)
            task = batch.get("task", {"pad_mask_dict": {}})
            pad_mask = obs["timestep_pad_mask"]

            guide_input = None
            if cfg.use_guidance:
                guide_input = flat(batch).get(cfg.guide_key)
                if guide_input is not None and guide_input.ndim > 3:
                    guide_input = guide_input.reshape(*guide_input.shape[:2], -1)

            actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h)

        with timer("train"):
            state, update_info = train_step(
                state,
                obs,
                task,
                pad_mask,
                actions,
                dof_ids,
                chunk_steps,
                guide_input=guide_input,
            )
        timer.tock("total")
        update_info = jax.device_get(update_info)
        total_loss = float(update_info["loss"])
        losses.append(total_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            embody_metrics = {}
            if "embody" in batch.get("act", {}):
                embody_metrics = per_embodiment_metrics(batch, update_info)
            print(f"\n[bold]step={step} loss={total_loss}:[/]")
            if embody_metrics:
                for k, v in sorted(embody_metrics.items()):
                    print(f"  {k}: {v:.4f}")
            row = [str(step), f"{total_loss:.4f}"]
            row.append(f"{float(update_info['grad_norm']):.4f}")
            table.add_row(*row)
            cfg.wandb.log(
                {
                    "training": update_info,
                    "timer": timer.get_average_times(),
                    **embody_metrics,
                },
                step=step,
            )

    print(table)

    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    ratio = last / first if first > 0 else float("inf")
    print(f"\nloss: {first:.4f} -> {last:.4f}  ({ratio:.2%} of initial)")
    cfg.wandb.log({"summary": {"loss_first": first, "loss_last": last, "loss_ratio": ratio}}, step=cfg.steps - 1)

    if ratio < 0.5:
        print("[bold green]loss decreased — training works[/]")
    else:
        print("[bold yellow]loss did not decrease much — check lr or architecture[/]")

    # -- denoise demo: Euler ODE solve -----------------------------------------
    print(Rule("predict_action: full denoise"))

    batch = next(dsit)
    obs = normalize_obs(batch["observation"], obs_keys)
    task = batch.get("task", {"pad_mask_dict": {}})

    params = state.model.params
    model_params = params["model"] if guide_module is not None else params
    bound = model.module.bind({"params": model_params})
    transformer_outputs = bound.crossformer_transformer(
        obs,
        task,
        obs["timestep_pad_mask"],
        train=False,
    )

    # Encode guidance tokens for inference
    guide_tokens = None
    if guide_module is not None:
        guide_eval = flat(batch).get(cfg.guide_key)
        if guide_eval is not None:
            if guide_eval.ndim > 3:
                guide_eval = guide_eval.reshape(*guide_eval.shape[:2], -1)
            guide_tokens = guide_module.apply(
                {"params": params["guide"]},
                guide_eval,
                deterministic=True,
            )

    actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h)
    n_valid = int((dof_ids[0] != 0).sum())  # non-MASK DOFs in first sample

    pred = bound.heads["xflow"].predict_action(
        transformer_outputs,
        rng=pred_rng,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        train=False,
        guidance_tokens=guide_tokens,
    )  # (B, W, max_h, max_a)


    # Compute MSE on valid region
    q_mask = build_query_mask(chunk_steps, dof_ids)
    pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
    tgt_flat = actions.reshape(actions.shape[0], actions.shape[1], -1)
    mask = jnp.broadcast_to(q_mask[:, None, :], pred_flat.shape)
    sq_err = (pred_flat - tgt_flat) ** 2 * mask
    mse = sq_err.sum() / mask.sum()

    pred_valid = pred[0, 0, :max_h, :n_valid]
    tgt_valid = actions[0, 0, :max_h, :n_valid]

    print(f"\n  pred shape: {pred.shape}  valid DOFs: {n_valid}")
    print(f"  mse (valid): {float(mse):.4f}")
    print(f"  pred range:  [{float(pred_valid.min()):.3f}, {float(pred_valid.max()):.3f}]")
    print(f"  tgt  range:  [{float(tgt_valid.min()):.3f}, {float(tgt_valid.max()):.3f}]")
    cfg.wandb.log(
        {
            "predict_action": {
                "mse": float(mse),
                "pred_min": float(pred_valid.min()),
                "pred_max": float(pred_valid.max()),
                "tgt_min": float(tgt_valid.min()),
                "tgt_max": float(tgt_valid.max()),
            }
        },
        step=cfg.steps,
    )


    # -- RTC demo -------------------------------------------------------------
    print(Rule("rtc_predict_action: guided denoise"))

    # Sahte önceki chunk olarak mevcut actions'ı kullan
    d_rtc = 4   # inference delay (adım cinsinden)
    s_rtc = 8   # execution horizon — d <= s <= max_h - d

    rtc_pred = rtc_predict_action(
        bound=bound,
        transformer_outputs=transformer_outputs,
        rng=pred_rng,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        a_prev=actions,          # (B, W, max_h, max_a) — önceki chunk
        d=d_rtc,
        s=s_rtc,
        beta=5.0,
        guidance_tokens=guide_tokens,
    )  # (B, W, max_h, max_a)

    rtc_pred_flat = rtc_pred.reshape(rtc_pred.shape[0], rtc_pred.shape[1], -1)
    rtc_sq_err = (rtc_pred_flat - tgt_flat) ** 2 * mask
    rtc_mse = rtc_sq_err.sum() / mask.sum()

    rtc_valid = rtc_pred[0, 0, :max_h, :n_valid]
    print(f"\n  rtc pred shape: {rtc_pred.shape}  valid DOFs: {n_valid}")
    print(f"  rtc mse (valid): {float(rtc_mse):.4f}")
    print(f"  rtc pred range:  [{float(rtc_valid.min()):.3f}, {float(rtc_valid.max()):.3f}]")
    cfg.wandb.log(
        {
            "rtc_predict_action": {
                "mse": float(rtc_mse),
                "pred_min": float(rtc_valid.min()),
                "pred_max": float(rtc_valid.max()),
            }
        },
        step=cfg.steps,
    )






    print("\n[bold green]done.[/]")
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
