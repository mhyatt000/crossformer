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

from dataclasses import dataclass
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

from crossformer.embody import slot_positions
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

    steps: int = 200  # training steps
    lr: float = 1e-3  # learning rate
    log_every: int = 10  # log interval
    batch_size: int = 4  # global batch size
    mix: str = "xgym_sweep"  # dataset mix name
    heads: tuple[str, ...] = ("single", "k3ds")  # action head keys to train on
    horizon: int = 20  # action horizon from data pipeline
    transformer_size: str = "dummy"  # transformer size preset
    obs_keys: tuple[str, ...] = ("proprio_.*", "time", "timestep")  # lowdim obs keys to tokenize


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


def make_loss_fn(module):
    """Build loss function: transformer → XFlowHead.loss for one head."""

    def loss_fn(params, obs, task, pad_mask, actions, dof_ids, chunk_steps, slot_pos, emb_mask, rng, train=True):
        bound = module.bind({"params": params}, rngs={"dropout": rng})
        transformer_outputs = bound.crossformer_transformer(
            obs,
            task,
            pad_mask,
            train=train,
        )
        loss, metrics = bound.heads["xflow"].loss(
            transformer_outputs,
            actions,
            dof_ids,
            chunk_steps,
            slot_pos=slot_pos,
            train=train,
        )
        frac = emb_mask.mean()
        return loss * frac, {k: v * frac for k, v in metrics.items()}

    return loss_fn


@partial(jax.jit, static_argnames=("loss_fn", "train"))
def train_step_single(
    state, loss_fn, obs, task, pad_mask, actions, dof_ids, chunk_steps, slot_pos, emb_mask, train=True
):
    """Forward + backward for one head, returns loss + grads."""
    rng = jax.random.fold_in(state.rng, state.step)

    def _loss(params):
        return loss_fn(params, obs, task, pad_mask, actions, dof_ids, chunk_steps, slot_pos, emb_mask, rng, train=train)

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    return loss, metrics, grads


@jax.jit
def apply_grads(state, grads):
    state = state.apply_gradients(grads=grads)
    grad_norm = optax.global_norm(grads)
    return state, grad_norm


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

    # Load data
    print(Rule("loading data"))
    import crossformer.cn as cn
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
    dataset = GrainDataFactory(mp=0).make(train_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=True)
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

    # Optimizer + state
    tx = optax.adamw(cfg.lr, weight_decay=1e-4)
    state = TrainStateRng.create(
        apply_fn=model.module.apply,
        params=model.params,
        tx=tx,
        rng=train_rng,
    )
    loss_fn = make_loss_fn(model.module)

    # Train
    print(Rule("training"))
    table = Table(title="training")
    table.add_column("step", justify="right", style="cyan")
    table.add_column("loss", justify="right")
    for h in cfg.heads:
        table.add_column(f"{h}", justify="right")
    table.add_column("|grad|", justify="right", style="dim")

    losses = []
    for step in tqdm(range(cfg.steps)):
        batch = next(dsit)
        obs = normalize_obs(batch["observation"], obs_keys)
        task = batch.get("task", {"pad_mask_dict": {}})
        pad_mask = obs["timestep_pad_mask"]

        total_loss = 0.0
        total_grads = None
        head_losses = {}

        for h in cfg.heads:
            emb_name = head_info[h]["embodiment"]
            inputs = prepare_head_inputs(
                batch,
                h,
                max_h,
                max_a,
                emb_name,
            )
            if inputs is None:
                continue
            actions, dof_ids, chunk_steps, slot_pos, emb_mask = inputs
            loss_h, _metrics_h, grads_h = train_step_single(
                state,
                loss_fn,
                obs,
                task,
                pad_mask,
                actions,
                dof_ids,
                chunk_steps,
                slot_pos,
                emb_mask,
            )
            total_loss += float(loss_h)
            head_losses[h] = float(loss_h)
            total_grads = grads_h if total_grads is None else jax.tree.map(lambda a, b: a + b, total_grads, grads_h)

        if total_grads is None:
            raise ValueError(f"No configured heads found in batch. wanted={cfg.heads} got={tuple(batch['action'])}")

        state, grad_norm = apply_grads(state, total_grads)
        losses.append(total_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"\n[bold]step={step} loss={total_loss}:[/]")
            row = [str(step), f"{total_loss:.4f}"]
            row.extend(f"{head_losses[h]:.4f}" for h in cfg.heads)
            row.append(f"{float(grad_norm):.4f}")
            table.add_row(*row)

    print(table)

    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    ratio = last / first if first > 0 else float("inf")
    print(f"\nloss: {first:.4f} -> {last:.4f}  ({ratio:.2%} of initial)")

    if ratio < 0.5:
        print("[bold green]loss decreased — training works[/]")
    else:
        print("[bold yellow]loss did not decrease much — check lr or architecture[/]")

    # -- denoise demo: Euler ODE solve per head --------------------------------
    print(Rule("predict_action: full denoise per head"))

    batch = next(dsit)
    obs = normalize_obs(batch["observation"], obs_keys)
    task = batch.get("task", {"pad_mask_dict": {}})

    bound = model.module.bind({"params": state.params})
    transformer_outputs = bound.crossformer_transformer(
        obs,
        task,
        obs["timestep_pad_mask"],
        train=False,
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

    print("\n[bold green]done.[/]")


if __name__ == "__main__":
    main(tyro.cli(Config))
