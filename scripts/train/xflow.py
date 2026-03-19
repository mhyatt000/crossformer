"""Standalone training loop for XFlowHead with real data.

Loads data via GrainDataFactory, extracts actions for each head,
maps to dof_ids/chunk_steps, and trains a shared XFlowHead on all heads.
Transformer outputs are zeros (no backbone).

Usage:
    uv run scripts/train/xflow.py
    uv run scripts/train/xflow.py --steps 500 --lr 3e-4
    uv run scripts/train/xflow.py --mix xgym_sweep --batch-size 4
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from rich import print
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm
import tyro

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.dof import (
    build_query_mask,
    chunk_range,
    EMBODIMENTS,
    pad_chunk_steps,
    pad_dof_ids,
    pad_slot_positions,
    slot_positions,
)
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.utils.spec import spec

# -- embodiment mapping -------------------------------------------------------

# Map batch action keys → DOF recipes in EMBODIMENTS.
HEAD_TO_EMBODIMENT = {
    "single": "xarm_gripper",
    "single_arm": "xarm_gripper",
    "mano": "mano",
    "k3ds": "k3ds",
}

# -- config -------------------------------------------------------------------

N_TOKENS = 8  # dummy transformer token count
EMBED_DIM = 512


@dataclass
class Config:
    """XFlowHead standalone training config."""

    steps: int = 200  # training steps
    lr: float = 1e-3  # learning rate
    log_every: int = 10  # log interval
    batch_size: int = 4  # global batch size
    mix: str = "xgym_sweep"  # dataset mix name
    heads: tuple[str, ...] = ("single", "k3ds")  # action head keys to train on
    horizon: int = 20  # action horizon from data pipeline


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


def make_head(max_h, max_a):
    return XFlowHead(
        readout_key="readout",
        max_dofs=max_a,
        max_horizon=max_h,
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
        flow_steps=10,
    )


def batch_to_xflow(batch, head_key, max_h, max_a, embodiment_name):
    """Extract actions for one head and build XFlowHead inputs."""
    actions_real = batch["action"][head_key]
    # Flatten multi-dim DOFs like k3ds (B, H, 21, 4) → (B, H, 84)
    if actions_real.ndim == 4 and actions_real.shape[-1] != actions_real.shape[-2]:
        # Heuristic: if last two dims are unequal and ndim=4, it's (B, H, joints, coords)
        actions_real = actions_real.reshape(*actions_real.shape[:2], -1)
    if actions_real.ndim == 3:
        actions_real = actions_real[:, None, :, :]
    B, W, H_real, A_real = actions_real.shape

    pad_h = max_h - H_real
    pad_a = max_a - A_real
    actions = jnp.pad(actions_real, ((0, 0), (0, 0), (0, pad_h), (0, pad_a)))

    dof_recipe = EMBODIMENTS[embodiment_name]
    dof_padded = jnp.array(pad_dof_ids(dof_recipe, max_a))
    chunk_padded = jnp.array(pad_chunk_steps(chunk_range(H_real), max_h))
    slot_padded = jnp.array(pad_slot_positions(slot_positions(len(dof_recipe)), max_a))
    dof_ids = jnp.tile(dof_padded[None], (B, 1))
    chunk_steps = jnp.tile(chunk_padded[None], (B, 1))
    slot_pos = jnp.tile(slot_padded[None], (B, 1))

    # Dummy transformer outputs (no backbone)
    tokens = jnp.zeros((B, W, N_TOKENS, EMBED_DIM))
    mask = jnp.ones((B, W, N_TOKENS), dtype=jnp.int32)
    transformer_outputs = {"readout": TokenGroup(tokens=tokens, mask=mask)}

    # Embodiment mask: which samples have data for this head
    emb_mask = batch["embodiment"].get(head_key)
    emb_mask = emb_mask.reshape(B) if emb_mask is not None else jnp.ones(B, dtype=jnp.bool_)

    return transformer_outputs, actions, dof_ids, chunk_steps, slot_pos, emb_mask


# -- train step ---------------------------------------------------------------


@partial(jax.jit, static_argnames=("train",))
def train_step_single(state, rng, transformer_outputs, actions, dof_ids, chunk_steps, slot_pos, emb_mask, train=True):
    """Train step for one head, returns masked loss."""
    dropout_rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        loss, metrics = state.apply_fn(
            {"params": params},
            transformer_outputs,
            actions,
            dof_ids,
            chunk_steps,
            slot_pos=slot_pos,
            train=train,
            method=XFlowHead.loss,
            rngs={"dropout": dropout_rng},
        )
        # Mask loss by embodiment (samples without this head contribute 0)
        frac = emb_mask.mean()
        return loss * frac, {k: v * frac for k, v in metrics.items()}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return loss, metrics, grads


@partial(jax.jit)
def apply_grads(state, grads):
    state = state.apply_gradients(grads=grads)
    grad_norm = optax.global_norm(grads)
    return state, grad_norm


# -- main ---------------------------------------------------------------------


def main(cfg: Config):
    print(Rule("XFlowHead: multi-head training on real data", style="bold magenta"))

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
    dataset = GrainDataFactory(mp=0).make(train_cfg, shard_fn=None, train=True)
    dsit = iter(dataset.dataset)
    example_batch = next(dsit)
    print(spec(example_batch))

    for h in cfg.heads:
        act = example_batch["action"][h]
        print(f"  action['{h}'] shape: {act.shape}")

    # Init model
    print(Rule("init model"))
    head = make_head(max_h, max_a)
    rng = jax.random.PRNGKey(42)
    init_rng, train_rng, pred_rng = jax.random.split(rng, 3)

    dummy = batch_to_xflow(example_batch, cfg.heads[0], max_h, max_a, head_info[cfg.heads[0]]["embodiment"])
    variables = head.init({"params": init_rng, "dropout": init_rng}, dummy[0])
    n_params = sum(x.size for x in jax.tree.leaves(variables["params"]))
    print(f"params: {n_params:,}")

    tx = optax.adamw(cfg.lr, weight_decay=1e-4)
    state = TrainState.create(apply_fn=head.apply, params=variables["params"], tx=tx)

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

        total_loss = 0.0
        total_grads = None
        head_losses = {}

        for h in cfg.heads:
            emb_name = head_info[h]["embodiment"]
            tout, actions, dof_ids, chunk_steps, slot_pos, emb_mask = batch_to_xflow(
                batch,
                h,
                max_h,
                max_a,
                emb_name,
            )
            loss_h, _metrics_h, grads_h = train_step_single(
                state,
                train_rng,
                tout,
                actions,
                dof_ids,
                chunk_steps,
                slot_pos,
                emb_mask,
            )
            total_loss += float(loss_h)
            head_losses[h] = float(loss_h)
            total_grads = (
                grads_h
                if total_grads is None
                else jax.tree.map(
                    lambda a, b: a + b,
                    total_grads,
                    grads_h,
                )
            )

        state, grad_norm = apply_grads(state, total_grads)
        losses.append(total_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
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
    for i, h in enumerate(cfg.heads):
        emb_name = head_info[h]["embodiment"]
        n_dofs = head_info[h]["n_dofs"]
        tout, actions, dof_ids, chunk_steps, slot_pos, _ = batch_to_xflow(
            batch,
            h,
            max_h,
            max_a,
            emb_name,
        )

        pred = head.apply(
            {"params": state.params},
            tout,
            rng=jax.random.fold_in(pred_rng, i),
            dof_ids=dof_ids,
            chunk_steps=chunk_steps,
            slot_pos=slot_pos,
            train=False,
            method=XFlowHead.predict_action,
        )  # (B, W, max_h, max_a)

        # Extract valid region and compute MSE
        q_mask = build_query_mask(chunk_steps, dof_ids, slot_pos)  # (B, max_h*max_a)
        pred_flat = pred.reshape(cfg.batch_size, 1, -1)
        tgt_flat = actions.reshape(cfg.batch_size, 1, -1)
        sq_err = (pred_flat - tgt_flat) ** 2 * q_mask[:, None, :]
        mse = sq_err.sum() / (q_mask.sum() * 1)  # avg over valid queries

        # Extract valid actions for display (first sample)
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
