"""End-to-end XFlowHead training: CrossFormerModel + real data.

Full transformer backbone with XFlowHead, trained on real data from
GrainDataFactory. Multi-head cross-embodiment training with per-head
gradient accumulation.

Usage:
    uv run scripts/train/xflow.py
    uv run scripts/train/xflow.py --steps 500 --lr 3e-4
    uv run scripts/train/xflow.py --mix xgym_sweep --batch-size 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import re

from flax.training.train_state import TrainState
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
from rich import print
from rich.rule import Rule
from rich.table import Table
import tensorflow as tf
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.embody import slot_positions
from crossformer.model.components.guidance import TokenGuidance
from crossformer.model.components.heads.dof import (
    build_query_mask,
    chunk_range,
    EMBODIMENTS,
    pad_chunk_steps,
    pad_dof_ids,
    pad_slot_positions,
)
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.tokenizers import LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.train_utils import Timer
from crossformer.utils.tree.core import flat
import wandb

# -- embodiment mapping -------------------------------------------------------

HEAD_TO_EMBODIMENT = {
    "single": "xarm_gripper",
    "single_arm": "xarm_gripper",
    "mano": "mano",
    "k3ds": "k3ds",
}

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
    heads: tuple[str, ...] = ("single", "k3ds")  # action head keys to train on
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


def resolve_heads(heads):
    """Resolve head names → embodiment recipes, compute max dims."""
    info = {}
    for h in heads:
        emb_name = HEAD_TO_EMBODIMENT.get(h)
        if emb_name is None:
            raise ValueError(f"No embodiment mapping for head '{h}'. Known: {list(HEAD_TO_EMBODIMENT)}")
        info[h] = {"embodiment": emb_name, "n_dofs": len(EMBODIMENTS[emb_name])}
    return info


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


def prepare_head_inputs(batch, head_key, max_h, max_a, embodiment_name):
    """Extract and pad actions + embodiment metadata for one head."""
    if head_key not in batch["action"]:
        return None

    actions_real = batch["action"][head_key]
    # Flatten multi-dim DOFs like k3ds (B, H, 21, 4) → (B, H, 84)
    if actions_real.ndim == 4 and actions_real.shape[-1] != actions_real.shape[-2]:
        actions_real = actions_real.reshape(*actions_real.shape[:2], -1)
    if actions_real.ndim == 3:
        actions_real = actions_real[:, None, :, :]
    B, W, H_real, A_real = actions_real.shape  # noqa RUF
    if H_real > max_h or A_real > max_a:
        raise ValueError(f"{head_key}: action shape {(H_real, A_real)} exceeds bounds {(max_h, max_a)}")

    actions = jnp.pad(actions_real, ((0, 0), (0, 0), (0, max_h - H_real), (0, max_a - A_real)))

    dof_recipe = EMBODIMENTS[embodiment_name]
    dof_ids = jnp.tile(jnp.array(pad_dof_ids(dof_recipe, max_a))[None], (B, 1))
    chunk_steps = jnp.tile(jnp.array(pad_chunk_steps(chunk_range(H_real), max_h))[None], (B, 1))
    slot_pos = jnp.tile(jnp.array(pad_slot_positions(slot_positions(len(dof_recipe)), max_a))[None], (B, 1))

    emb_mask = batch["embodiment"].get(head_key)
    emb_mask = emb_mask.reshape(B) if emb_mask is not None else jnp.ones(B, dtype=jnp.bool_)

    return actions, dof_ids, chunk_steps, slot_pos, emb_mask


def shard_batch(batch, mesh):
    """Shard a host-local batch across the data axis."""
    return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))


# -- TrainState with RNG -----------------------------------------------------


class TrainStateRng(TrainState):
    rng: jax.Array

    def apply_gradients(self, *, grads, **kwargs):
        _, new_rng = jax.random.split(self.rng)
        state = super().apply_gradients(grads=grads, **kwargs)
        return state.replace(rng=new_rng)


# -- train / eval steps ------------------------------------------------------


def make_train_step(module, head_names, lr, guide_module=None):
    """Build a fully compiled multi-head train step.

    Runs the transformer once, then computes loss + grads for each head
    inside a single JIT-compiled function via jax.lax.scan over stacked
    per-head inputs.

    Args:
        module: CrossFormerModel module.
        head_names: action heads to report metrics for.
        lr: optimizer learning rate.
        guide_module: optional TokenGuidance module.

    Returns:
        Compiled train_step(state, obs, task, pad_mask, head_inputs, guide_input).
    """

    def _loss_one_head(params, transformer_outputs, head_input, guidance_tokens, rng, train):
        """Loss for a single head given precomputed transformer outputs."""
        actions, dof_ids, chunk_steps, slot_pos, emb_mask = head_input
        bound = module.bind({"params": params}, rngs={"dropout": rng})
        loss, metrics = bound.heads["xflow"].loss(
            transformer_outputs,
            actions,
            dof_ids,
            chunk_steps,
            slot_pos=slot_pos,
            train=train,
            guidance_tokens=guidance_tokens,
        )
        frac = emb_mask.mean()
        metrics = {k: v * frac for k, v in metrics.items()}
        return loss * frac, metrics

    @partial(jax.jit, static_argnames=("train",))
    def train_step(state, obs, task, pad_mask, head_inputs, guide_input=None, train=True):
        """Full multi-head train step: fwd transformer once, loss per head, sum grads.

        Args:
            state: TrainStateRng (params are {"model": ..., "guide": ...} or flat).
            obs: observation dict.
            task: task dict.
            pad_mask: (B, W) timestep pad mask.
            head_inputs: tuple of (actions, dof_ids, chunk_steps, slot_pos, emb_mask)
                per head — each array has shape (B, ...).
            guide_input: optional (B, S, D) guidance signal.
            train: bool.

        Returns:
            (state, update_info).
        """
        rng = jax.random.fold_in(state.rng, state.step)

        def _total_loss(params):
            model_params = params["model"] if guide_module is not None else params

            bound = module.bind({"params": model_params}, rngs={"dropout": rng})
            transformer_outputs = bound.crossformer_transformer(obs, task, pad_mask, train=train)

            guidance_tokens = None
            if guide_module is not None and guide_input is not None:
                guidance_tokens = guide_module.apply(
                    {"params": params["guide"]},
                    guide_input,
                    deterministic=not train,
                )

            total = jnp.float32(0.0)
            head_losses = []
            head_metrics = []
            for hi in head_inputs:
                loss_h, metrics_h = _loss_one_head(model_params, transformer_outputs, hi, guidance_tokens, rng, train)
                total = total + loss_h
                head_losses.append(loss_h)
                head_metrics.append(metrics_h)

            return total, (jnp.stack(head_losses), tuple(head_metrics))

        (total_loss, (head_losses, head_metrics)), grads = jax.value_and_grad(_total_loss, has_aux=True)(state.params)
        updates, _ = state.tx.update(grads, state.opt_state, state.params)
        update_info = {
            "total_loss": total_loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": optax.global_norm(state.params),
            "learning_rate": jnp.asarray(lr),
        }
        for head_name, head_loss, metrics in zip(head_names, head_losses, head_metrics):
            update_info[head_name] = {"weighted_loss": head_loss, **metrics}
        state = state.apply_gradients(grads=grads)
        return state, update_info

    return train_step


# -- main ---------------------------------------------------------------------


def main(cfg: Config):
    tf.config.set_visible_devices([], "GPU")
    initialize_compilation_cache()
    devices = jax.devices()
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    print(Rule("XFlowHead + CrossFormerModel: real data", style="bold magenta"))
    print(f"  backend={jax.default_backend()} devices={len(devices)}")
    if cfg.batch_size % len(devices) != 0:
        raise ValueError(f"batch_size={cfg.batch_size} must be divisible by devices={len(devices)}")

    # Resolve embodiments and compute max dims
    head_info = resolve_heads(cfg.heads)
    max_a = max(info["n_dofs"] for info in head_info.values())
    max_h = cfg.horizon
    for h, info in head_info.items():
        print(f"  {h:15s} -> {info['embodiment']:20s}  dofs={info['n_dofs']}")
    print(f"  max_h={max_h}  max_a={max_a}")
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

    for h in cfg.heads:
        act = example_batch["action"][h]
        print(f"  action['{h}'] shape: {act.shape}")

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
            "head_info": head_info,
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
        # Flatten trailing dims if needed: (B, W, ...) → (B, W, D)
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
    state = TrainStateRng.create(
        apply_fn=model.module.apply,
        params=combined_params,
        tx=tx,
        rng=train_rng,
    )
    train_step = make_train_step(model.module, cfg.heads, cfg.lr, guide_module=guide_module)

    # Train
    print(Rule("training"))
    table = Table(title="training")
    table.add_column("step", justify="right", style="cyan")
    table.add_column("loss", justify="right")
    for h in cfg.heads:
        table.add_column(f"{h}", justify="right")
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

            head_inputs = []
            for h in cfg.heads:
                emb_name = head_info[h]["embodiment"]
                inputs = prepare_head_inputs(batch, h, max_h, max_a, emb_name)
                if inputs is None:
                    raise ValueError(f"Head '{h}' not found in batch. got={tuple(batch['action'])}")
                head_inputs.append(inputs)

        with timer("train"):
            state, update_info = train_step(
                state,
                obs,
                task,
                pad_mask,
                tuple(head_inputs),
                guide_input=guide_input,
            )
        timer.tock("total")
        update_info = jax.device_get(update_info)
        total_loss = float(update_info["total_loss"])
        losses.append(total_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"\n[bold]step={step} loss={total_loss}:[/]")
            row = [str(step), f"{total_loss:.4f}"]
            row.extend(f"{float(update_info[h]['weighted_loss']):.4f}" for h in cfg.heads)
            row.append(f"{float(update_info['grad_norm']):.4f}")
            table.add_row(*row)
            cfg.wandb.log(
                {
                    "training": update_info,
                    "timer": timer.get_average_times(),
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

    # -- denoise demo: Euler ODE solve per head --------------------------------
    print(Rule("predict_action: full denoise per head"))

    batch = next(dsit)
    obs = normalize_obs(batch["observation"], obs_keys)
    task = batch.get("task", {"pad_mask_dict": {}})

    model_params = state.params["model"] if guide_module is not None else state.params
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
                {"params": state.params["guide"]},
                guide_eval,
                deterministic=True,
            )

    for i, h in enumerate(cfg.heads):
        emb_name = head_info[h]["embodiment"]
        n_dofs = head_info[h]["n_dofs"]
        inputs = prepare_head_inputs(
            batch,
            h,
            max_h,
            max_a,
            emb_name,
        )
        if inputs is None:
            continue
        actions, dof_ids, chunk_steps, slot_pos, _ = inputs

        pred = bound.heads["xflow"].predict_action(
            transformer_outputs,
            rng=jax.random.fold_in(pred_rng, i),
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            slot_pos=slot_pos,
            train=False,
            guidance_tokens=guide_tokens,
        )  # (B, W, max_h, max_a)

        # Extract valid region and compute MSE
        q_mask = build_query_mask(chunk_steps, dof_ids, slot_pos)
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        tgt_flat = actions.reshape(actions.shape[0], actions.shape[1], -1)
        mask = jnp.broadcast_to(q_mask[:, None, :], pred_flat.shape)
        sq_err = (pred_flat - tgt_flat) ** 2 * mask
        mse = sq_err.sum() / mask.sum()

        pred_valid = pred[0, 0, : cfg.horizon, :n_dofs]
        tgt_valid = actions[0, 0, : cfg.horizon, :n_dofs]

        print(f"\n  [bold]{h}[/] ({emb_name}, {n_dofs} DOFs)")
        print(f"    pred shape: {pred.shape}")
        print(f"    mse (valid): {float(mse):.4f}")
        print(f"    pred range:  [{float(pred_valid.min()):.3f}, {float(pred_valid.max()):.3f}]")
        print(f"    tgt  range:  [{float(tgt_valid.min()):.3f}, {float(tgt_valid.max()):.3f}]")
        cfg.wandb.log(
            {
                "predict_action": {
                    h: {
                        "mse": float(mse),
                        "pred_min": float(pred_valid.min()),
                        "pred_max": float(pred_valid.max()),
                        "tgt_min": float(tgt_valid.min()),
                        "tgt_max": float(tgt_valid.max()),
                    }
                }
            },
            step=cfg.steps,
        )

    print("\n[bold green]done.[/]")
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
