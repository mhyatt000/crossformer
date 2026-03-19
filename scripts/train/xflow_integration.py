"""Integration test: CrossFormerModel with XFlowHead as the only head.

Builds a full CrossFormerModel (transformer + tokenizers + XFlowHead),
trains on synthetic mixed-embodiment batches, and verifies end-to-end
gradient flow from transformer backbone through the perceiver IO head.

Usage:
    uv run scripts/train/xflow_integration.py
    uv run scripts/train/xflow_integration.py --steps 100 --lr 1e-3
"""

from __future__ import annotations

from functools import partial

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from rich import print
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm

from crossformer.model.components.heads.dof import (
    build_query_mask,
    chunk_range,
    chunk_strided,
    EMBODIMENTS,
    pad_chunk_steps,
    pad_dof_ids,
)
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.tokenizers import LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.spec import ModuleSpec

# -- config -------------------------------------------------------------------

MAX_H = 20
MAX_A = 18
B = 4
W = 2  # window / horizon for transformer
PROPRIO_DIM = 10

RECIPES = [
    ("xarm_gripper", EMBODIMENTS["xarm_gripper"], chunk_range(10)),
    ("xarm_ruka", EMBODIMENTS["xarm_ruka"], chunk_range(20)),
    ("cartesian_pose", EMBODIMENTS["cartesian_pose"], chunk_strided(20, 2)),
    ("cartesian_pos", EMBODIMENTS["cartesian_pos"], chunk_strided(100, 25)),
]


# -- helpers ------------------------------------------------------------------


def pad_embodiment(dof_ids, steps):
    return (
        jnp.array(pad_dof_ids(dof_ids, MAX_A)),
        jnp.array(pad_chunk_steps(steps, MAX_H)),
    )


def make_model_config():
    """Build a minimal CrossFormerModel config with XFlowHead."""
    token_dim, transformer_kwargs = common_transformer_sizes("dummy")

    readout_name = "xflow"
    readout_key = f"readout_{readout_name}"
    config = {
        "model": {
            "observation_tokenizers": {
                "proprio": ModuleSpec.create(
                    LowdimObsTokenizer,
                    obs_keys=["proprio"],
                ),
            },
            "task_tokenizers": {},
            "heads": {
                "xflow": ModuleSpec.create(
                    XFlowHead,
                    readout_key=readout_key,
                    max_dofs=MAX_A,
                    max_horizon=MAX_H,
                    num_query_channels=128,
                    num_heads=4,
                    num_self_attend_layers=1,
                    flow_steps=10,
                ),
            },
            "readouts": {readout_name: 4},
            "token_embedding_size": token_dim,
            "transformer_kwargs": transformer_kwargs,
            "max_horizon": W,
        },
        "text_processor": None,
    }
    return config


def make_example_batch(rng):
    """Create a batch matching CrossFormerModel expectations.

    Returns (batch, dof_ids, chunk_steps) — the latter two are XFlowHead
    metadata not consumed by the standard ActionHead interface.
    """
    k1, k2, k3 = jax.random.split(rng, 3)

    # Observation data matching LowdimObsTokenizer(obs_keys=["proprio"])
    proprio = jax.random.normal(k1, (B, W, PROPRIO_DIM))
    timestep_pad_mask = jnp.ones((B, W), dtype=jnp.bool_)

    # Per-sample embodiment metadata
    dof_list, chunk_list = [], []
    for i in range(B):
        _, dof, steps = RECIPES[i % len(RECIPES)]
        d, c = pad_embodiment(dof, steps)
        dof_list.append(d)
        chunk_list.append(c)
    dof_ids = jnp.stack(dof_list)
    chunk_steps = jnp.stack(chunk_list)

    # Structured target actions: sinusoid so the head can learn
    q_mask = build_query_mask(chunk_steps, dof_ids)  # (B, H*A)
    t = jnp.linspace(0, 2 * jnp.pi, MAX_H * MAX_A)
    pattern = jnp.sin(t) * 0.5
    scale = jax.random.uniform(k2, (B, 1), minval=0.5, maxval=1.5)
    noise = jax.random.normal(k3, (B, W, MAX_H * MAX_A)) * 0.05
    actions_flat = scale[:, None, :] * pattern[None, None, :] + noise
    actions_flat = actions_flat * q_mask[:, None, :]
    actions = actions_flat.reshape(B, W, MAX_H, MAX_A)

    batch = {
        "observation": {
            "proprio": proprio,
            "timestep_pad_mask": timestep_pad_mask,
        },
        "task": {
            "pad_mask_dict": {},
        },
    }
    return batch, actions, dof_ids, chunk_steps


# -- train / eval steps -------------------------------------------------------


def make_loss_fn(model):
    """Build a loss function that calls the transformer + XFlowHead.loss."""

    def loss_fn(params, batch, actions, dof_ids, chunk_steps, rng, train=True):
        bound = model.module.bind({"params": params}, rngs={"dropout": rng})

        # Run transformer backbone
        transformer_outputs = bound.crossformer_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        # Call XFlowHead.loss directly (different signature from standard heads)
        head = bound.heads["xflow"]
        loss, metrics = head.loss(
            transformer_outputs,
            actions,
            dof_ids,
            chunk_steps,
            train=train,
        )
        return loss, metrics

    return loss_fn


@partial(jax.jit, static_argnames=("loss_fn", "train"))
def train_step(state, loss_fn, batch, actions, dof_ids, chunk_steps, train=True):
    dropout_rng = jax.random.fold_in(state.rng, state.step)

    def _loss(params):
        return loss_fn(params, batch, actions, dof_ids, chunk_steps, dropout_rng, train=train)

    (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
    grad_norm = optax.global_norm(grads)

    # Standard TrainState from flax doesn't have .rng, so we use custom state
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, metrics, grad_norm


# -- custom TrainState with rng -----------------------------------------------


class TrainStateRng(TrainState):
    rng: jax.Array

    def apply_gradients(self, *, grads, **kwargs):
        _rng, new_rng = jax.random.split(self.rng)
        state = super().apply_gradients(grads=grads, **kwargs)
        return state.replace(rng=new_rng)


# -- main ---------------------------------------------------------------------


def main(steps: int = 200, lr: float = 1e-3, log_every: int = 10):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--log-every", type=int, default=log_every)
    args = parser.parse_args()

    print(Rule("XFlowHead + CrossFormerModel Integration", style="bold magenta"))

    rng = jax.random.PRNGKey(42)
    init_rng, data_rng, train_rng = jax.random.split(rng, 3)

    # -- Build config and example batch --
    config = make_model_config()
    example_batch, _, _, _ = make_example_batch(data_rng)

    print("[bold]config:[/]")
    print(f"  transformer: dummy (embed={config['model']['token_embedding_size']})")
    print(f"  max_H={MAX_H}, max_A={MAX_A}, window={W}")

    # -- Init CrossFormerModel --
    print(Rule("initializing CrossFormerModel"))
    model = CrossFormerModel.from_config(
        config,
        example_batch,
        text_processor=None,
        verbose=False,
        rng=init_rng,
    )
    n_params = sum(x.size for x in jax.tree.leaves(model.params))
    print(f"  params: {n_params:,}")
    print(f"  heads: {list(model.module.heads.keys())}")

    # -- Optimizer --
    tx = optax.adamw(args.lr, weight_decay=1e-4)
    state = TrainStateRng.create(
        apply_fn=model.module.apply,
        params=model.params,
        tx=tx,
        rng=train_rng,
    )
    loss_fn = make_loss_fn(model)

    # -- Training loop --
    print(Rule("training"))
    table = Table(title="training")
    table.add_column("step", justify="right", style="cyan")
    table.add_column("loss", justify="right")
    table.add_column("mse", justify="right")
    table.add_column("|grad|", justify="right", style="dim")

    losses = []
    for step in tqdm(range(args.steps)):
        step_rng = jax.random.fold_in(data_rng, step)
        batch, actions, dof_ids, chunk_steps = make_example_batch(step_rng)

        state, loss, metrics, grad_norm = train_step(
            state,
            loss_fn,
            batch,
            actions,
            dof_ids,
            chunk_steps,
        )

        loss_val = float(loss)
        losses.append(loss_val)

        if step % args.log_every == 0 or step == args.steps - 1:
            table.add_row(
                str(step),
                f"{loss_val:.4f}",
                f"{float(metrics['mse']):.4f}",
                f"{float(grad_norm):.4f}",
            )

    print(table)

    # -- Summary --
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    ratio = last / first if first > 0 else float("inf")
    print(f"\nloss: {first:.4f} -> {last:.4f}  ({ratio:.2%} of initial)")

    if ratio < 0.5:
        print("[bold green]loss decreased — integration works[/]")
    else:
        print("[bold yellow]loss did not decrease much — check lr or architecture[/]")

    # -- Inference demo --
    print(Rule("predict_action (Euler ODE solve)"))
    pred_rng = jax.random.PRNGKey(99)
    batch, actions, dof_ids, chunk_steps = make_example_batch(pred_rng)

    # Run transformer, then predict_action on the head via bound module
    bound = model.module.bind({"params": state.params})
    transformer_outputs = bound.crossformer_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=False,
    )

    pred = bound.heads["xflow"].predict_action(
        transformer_outputs,
        rng=pred_rng,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        train=False,
    )

    q_mask = build_query_mask(chunk_steps, dof_ids)
    print(f"  pred shape: {pred.shape}")
    print(f"  pred range: [{float(pred.min()):.3f}, {float(pred.max()):.3f}]")

    # Per-sample masked MSE
    pred_flat = pred.reshape(B, W, MAX_H * MAX_A)
    tgt_flat = actions.reshape(B, W, MAX_H * MAX_A)
    sq_err = (pred_flat - tgt_flat) ** 2 * q_mask[:, None, :]
    per_sample = sq_err.sum(axis=(1, 2)) / q_mask.sum(axis=1, keepdims=True)[:, :W].sum(axis=1)
    for i, (name, _, _) in enumerate(RECIPES[:B]):
        print(f"  [{i}] {name:20s}  mse={float(per_sample[i]):.4f}")

    print("\n[bold green]done.[/]")


if __name__ == "__main__":
    main()
