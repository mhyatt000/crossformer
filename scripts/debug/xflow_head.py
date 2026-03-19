"""Forward pass demo for XFlowHead (single-kernel cross-embodiment training)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rich import print
from rich.rule import Rule

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.dof import (
    build_query_mask,
    chunk_range,
    chunk_strided,
    EMBODIMENTS,
    pad_chunk_steps,
    pad_dof_ids,
)
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.utils.spec import spec

# -- helpers ------------------------------------------------------------------

MAX_H = 20  # max chunk steps across all embodiments
MAX_A = 18  # max DOFs across all embodiments (xarm_ruka = 7+11)


def show(label, tree):
    if tree is None:
        print(f"  [dim]{label}[/]: None")
    elif hasattr(tree, "shape"):
        print(spec({label: tree}))
    else:
        print(spec(tree))


def param_count(variables):
    n = sum(x.size for x in jax.tree.leaves(variables["params"]))
    print(f"  [bold green]params[/]: {n:,}")


def make_transformer_outputs(rng, batch, window, n_tokens, embed_dim):
    tokens = jax.random.normal(rng, (batch, window, n_tokens, embed_dim))
    mask = jnp.ones((batch, window, n_tokens), dtype=jnp.int32)
    return {"readout": TokenGroup(tokens=tokens, mask=mask)}


def make_head():
    """Single head instance shared across all embodiments."""
    return XFlowHead(
        readout_key="readout",
        max_dofs=MAX_A,
        max_horizon=MAX_H,
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
        flow_steps=10,
    )


def pad_embodiment(dof_ids, steps):
    """Pad a single embodiment's specs to max size."""
    return (
        jnp.array(pad_dof_ids(dof_ids, MAX_A)),
        jnp.array(pad_chunk_steps(steps, MAX_H)),
    )


# -- demos --------------------------------------------------------------------


def run_single_embodiment():
    """Single embodiment forward pass."""
    print(Rule("single embodiment: xarm_gripper"))
    B, W, N, E = 4, 2, 8, 512
    head = make_head()

    rng = jax.random.PRNGKey(0)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)
    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    # Pad to max size
    dof_padded, chunk_padded = pad_embodiment(
        EMBODIMENTS["xarm_gripper"],
        chunk_range(10),
    )
    dof_ids = jnp.tile(dof_padded[None], (B, 1))  # (B, MAX_A)
    chunk_steps = jnp.tile(chunk_padded[None], (B, 1))  # (B, MAX_H)

    time = jax.random.uniform(fwd_rng, (B, W, 1))
    a_t = jax.random.normal(fwd_rng, (B, W, MAX_H, MAX_A))

    pred = head.apply(
        variables,
        outputs,
        time=time,
        a_t=a_t,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        train=False,
        rngs={"dropout": fwd_rng},
    )

    q_mask = build_query_mask(chunk_steps, dof_ids)
    valid = q_mask[0].sum()
    total = MAX_H * MAX_A

    print("[bold]io:[/]")
    show("pred", pred)
    print(f"  valid queries: {int(valid)} / {total} (H=10 x A=8 = 80)")
    param_count(variables)


def run_mixed_batch():
    """Mixed batch: different embodiments in the same forward pass (1 kernel)."""
    print(Rule("mixed batch: 4 embodiments, 1 forward pass"))
    B, W, N, E = 4, 2, 8, 512
    head = make_head()

    rng = jax.random.PRNGKey(1)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)
    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    # 4 different embodiments in one batch
    embodiments = [
        ("xarm_gripper", EMBODIMENTS["xarm_gripper"], chunk_range(10)),
        ("xarm_ruka", EMBODIMENTS["xarm_ruka"], chunk_range(20)),
        ("cartesian_pose", EMBODIMENTS["cartesian_pose"], chunk_strided(20, 2)),
        ("cartesian_pos", EMBODIMENTS["cartesian_pos"], chunk_strided(100, 25)),
    ]

    dof_list, chunk_list = [], []
    for _, dof, steps in embodiments:
        d, c = pad_embodiment(dof, steps)
        dof_list.append(d)
        chunk_list.append(c)
    dof_ids = jnp.stack(dof_list)  # (4, MAX_A)
    chunk_steps = jnp.stack(chunk_list)  # (4, MAX_H)

    time = jax.random.uniform(fwd_rng, (B, W, 1))
    a_t = jax.random.normal(fwd_rng, (B, W, MAX_H, MAX_A))

    pred = head.apply(
        variables,
        outputs,
        time=time,
        a_t=a_t,
        dof_ids=dof_ids,
        chunk_steps=chunk_steps,
        train=False,
        rngs={"dropout": fwd_rng},
    )

    print("[bold]mixed batch — per-sample valid queries:[/]")
    q_mask = build_query_mask(chunk_steps, dof_ids)
    for i, (name, dof, steps) in enumerate(embodiments):
        valid = int(q_mask[i].sum())
        print(f"  [{i}] {name:20s}  H={len(steps):2d} x A={len(dof):2d} = {valid:3d} valid / {MAX_H * MAX_A}")

    show("pred", pred)
    param_count(variables)


def run_loss():
    """Loss computation with mixed embodiments."""
    print(Rule("loss: mixed embodiments"))
    B, W, N, E = 4, 2, 8, 512
    head = make_head()

    rng = jax.random.PRNGKey(2)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)
    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    # Two different embodiments repeated
    dof_a, chunk_a = pad_embodiment(EMBODIMENTS["xarm_gripper"], chunk_range(10))
    dof_b, chunk_b = pad_embodiment(EMBODIMENTS["cartesian_pose"], chunk_strided(20, 2))
    dof_ids = jnp.stack([dof_a, dof_b, dof_a, dof_b])
    chunk_steps = jnp.stack([chunk_a, chunk_b, chunk_a, chunk_b])

    actions = jax.random.normal(fwd_rng, (B, W, MAX_H, MAX_A))

    loss, metrics = head.apply(
        variables,
        outputs,
        actions,
        dof_ids,
        chunk_steps,
        train=True,
        method=head.loss,
        rngs={"dropout": fwd_rng},
    )

    print(f"  loss: {float(loss):.6f}")
    print(f"  mse:  {float(metrics['mse']):.6f}")
    print(f"  sign: {float(metrics['lsign']):.6f}")
    param_count(variables)


def main():
    print(Rule("XFlowHead: Single-Kernel Cross-Embodiment Demo", style="bold magenta"))
    run_single_embodiment()
    run_mixed_batch()
    run_loss()
    print("\n[bold green]All demos completed successfully.[/]")


if __name__ == "__main__":
    main()
