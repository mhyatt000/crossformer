"""Forward pass demo for XFlowHead (Perceiver IO flow-matching action head)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rich import print
from rich.rule import Rule

from crossformer.model.components.base import TokenGroup
from crossformer.model.components.heads.dof import (
    chunk_range,
    chunk_strided,
    EMBODIMENTS,
    ids,
)
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.utils.spec import spec


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


def make_transformer_outputs(rng, batch, window, n_tokens, embed_dim, readout_key="readout"):
    """Simulate transformer output tokens."""
    tokens = jax.random.normal(rng, (batch, window, n_tokens, embed_dim))
    mask = jnp.ones((batch, window, n_tokens), dtype=jnp.int32)
    return {readout_key: TokenGroup(tokens=tokens, mask=mask)}


def run_init_and_forward():
    """Basic init + forward pass with factored queries."""
    print(Rule("XFlowHead: init + forward (xarm_gripper)"))
    B, W, N, E = 4, 2, 8, 512  # batch, window, n_tokens, embed_dim
    H = 20  # action_horizon
    dof = EMBODIMENTS["xarm_gripper"]  # (j0..j6, gripper) → 8 DOFs

    head = XFlowHead(
        readout_key="readout",
        dof_ids=dof,
        chunk_steps=chunk_range(H),
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
        flow_steps=10,
    )

    rng = jax.random.PRNGKey(0)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)

    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    A = len(dof)
    time = jax.random.uniform(fwd_rng, (B, W, 1))
    a_t = jax.random.normal(fwd_rng, (B, W, H, A))
    pred = head.apply(
        variables,
        outputs,
        time=time,
        a_t=a_t,
        train=False,
        rngs={"dropout": fwd_rng},
    )

    print("[bold]transformer tokens:[/]")
    show("tokens", outputs["readout"].tokens)
    print("[bold]flow inputs:[/]")
    show("time", time)
    show("a_t", a_t)
    print("[bold]output:[/]")
    show("pred", pred)
    print(f"  expected: ({B}, {W}, {H * A})")
    print(f"  queries per step: {H} chunks x {A} DOFs = {H * A}")
    param_count(variables)


def run_various_embodiments():
    """Show DOF vocab sharing: different embodiments, shared embeddings."""
    print(Rule("XFlowHead: various embodiments (DOF vocab)"))
    B, W, N, E = 2, 1, 4, 256
    rng = jax.random.PRNGKey(1)

    configs = [
        {"dof": EMBODIMENTS["xarm_gripper"], "H": 4, "label": "xarm+gripper (7+1), H=4"},
        {"dof": EMBODIMENTS["xarm"], "H": 4, "label": "xarm no gripper (7), H=4"},
        {"dof": EMBODIMENTS["xarm_ruka"], "H": 20, "label": "xarm+ruka (7+11), H=20"},
        {"dof": EMBODIMENTS["cartesian_pose"], "H": 10, "label": "cartesian pose (6), H=10"},
        {"dof": EMBODIMENTS["cart_pose_gripper"], "H": 1, "label": "cart+grip (7), H=1"},
        {"dof": ids("ee_x", "ee_y", "ee_z"), "H": 50, "label": "cart position (3), H=50"},
    ]

    for cfg in configs:
        dof, H = cfg["dof"], cfg["H"]
        A = len(dof)
        head = XFlowHead(
            readout_key="readout",
            dof_ids=dof,
            chunk_steps=chunk_range(H),
            num_query_channels=128,
            num_heads=4,
            num_self_attend_layers=1,
        )
        init_rng, fwd_rng = jax.random.split(rng)
        rng = fwd_rng
        outputs = make_transformer_outputs(init_rng, B, W, N, E)
        variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

        time = jax.random.uniform(fwd_rng, (B, W, 1))
        a_t = jax.random.normal(fwd_rng, (B, W, H, A))
        pred = head.apply(
            variables,
            outputs,
            time=time,
            a_t=a_t,
            train=False,
            rngs={"dropout": fwd_rng},
        )
        n = sum(x.size for x in jax.tree.leaves(variables["params"]))
        print(f"  {cfg['label']:40s} -> pred {pred.shape}  queries={H * A:4d}  params={n:,}")


def run_strided_chunks():
    """Multi-resolution: coarse + dense chunk strategies."""
    print(Rule("XFlowHead: strided chunk steps (multi-resolution)"))
    B, W, N, E = 2, 1, 4, 256
    dof = EMBODIMENTS["xarm_gripper"]
    A = len(dof)
    rng = jax.random.PRNGKey(4)

    strategies = [
        ("dense H=20", chunk_range(20)),
        ("stride=5, end=50", chunk_strided(50, 5)),
        ("stride=2, end=20", chunk_strided(20, 2)),
        ("stride=10, end=100", chunk_strided(100, 10)),
    ]

    for label, steps in strategies:
        H = len(steps)
        head = XFlowHead(
            readout_key="readout",
            dof_ids=dof,
            chunk_steps=steps,
            num_query_channels=128,
            num_heads=4,
            num_self_attend_layers=1,
        )
        init_rng, fwd_rng = jax.random.split(rng)
        rng = fwd_rng
        outputs = make_transformer_outputs(init_rng, B, W, N, E)
        variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

        time = jax.random.uniform(fwd_rng, (B, W, 1))
        a_t = jax.random.normal(fwd_rng, (B, W, H, A))
        pred = head.apply(
            variables,
            outputs,
            time=time,
            a_t=a_t,
            train=False,
            rngs={"dropout": fwd_rng},
        )
        n = sum(x.size for x in jax.tree.leaves(variables["params"]))
        print(f"  {label:30s} steps={H:3d}  queries={H * A:4d}  pred={pred.shape}  params={n:,}")


def run_with_guidance():
    """Forward pass with classifier-free guidance tokens."""
    print(Rule("XFlowHead: guidance tokens (CFG)"))
    B, W, N, E = 4, 2, 8, 512
    dof = EMBODIMENTS["xarm_gripper"]
    H, A = 10, len(dof)
    G = 5

    head = XFlowHead(
        readout_key="readout",
        dof_ids=dof,
        chunk_steps=chunk_range(H),
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
    )

    rng = jax.random.PRNGKey(2)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)
    guidance_tokens = jax.random.normal(fwd_rng, (B, G, E))
    guidance_mask = jnp.ones((B, G), dtype=jnp.int32)
    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    time = jax.random.uniform(fwd_rng, (B, W, 1))
    a_t = jax.random.normal(fwd_rng, (B, W, H, A))

    pred_cond = head.apply(
        variables,
        outputs,
        time=time,
        a_t=a_t,
        train=False,
        guidance_tokens=guidance_tokens,
        guidance_mask=guidance_mask,
        rngs={"dropout": fwd_rng},
    )

    zero_mask = jnp.zeros((B, G), dtype=jnp.int32)
    pred_uncond = head.apply(
        variables,
        outputs,
        time=time,
        a_t=a_t,
        train=False,
        guidance_tokens=guidance_tokens,
        guidance_mask=zero_mask,
        rngs={"dropout": fwd_rng},
    )

    print("[bold]guidance:[/]")
    show("guidance_tokens", guidance_tokens)
    show("guidance_mask", guidance_mask)
    print("[bold]conditioned vs unconditioned:[/]")
    show("pred_cond", pred_cond)
    show("pred_uncond", pred_uncond)
    diff = jnp.abs(pred_cond - pred_uncond).mean()
    print(f"  mean |cond - uncond| = {float(diff):.6f}")
    print("  (non-zero means guidance tokens affect output)")

    cfg_scale = 2.0
    pred_cfg = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    show("pred_cfg (scale=2.0)", pred_cfg)
    param_count(variables)


def run_loss():
    """Compute flow matching loss."""
    print(Rule("XFlowHead: loss computation"))
    B, W, N, E = 4, 2, 8, 512
    dof = EMBODIMENTS["xarm_gripper"]
    H, A = 10, len(dof)

    head = XFlowHead(
        readout_key="readout",
        dof_ids=dof,
        chunk_steps=chunk_range(H),
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
    )

    rng = jax.random.PRNGKey(3)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)
    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    actions = jax.random.normal(fwd_rng, (B, W, H, A))
    timestep_pad_mask = jnp.ones((B, W), dtype=bool)
    action_pad_mask = jnp.ones((B, W, H, A), dtype=bool)

    loss, metrics = head.apply(
        variables,
        outputs,
        actions,
        timestep_pad_mask,
        action_pad_mask,
        train=True,
        method=head.loss,
        rngs={"dropout": fwd_rng},
    )

    print(f"  loss: {float(loss):.6f}")
    print(f"  mse:  {float(metrics['mse']):.6f}")
    print(f"  sign: {float(metrics['lsign']):.6f}")
    param_count(variables)


def main():
    print(Rule("XFlowHead Forward Pass Demo", style="bold magenta"))
    run_init_and_forward()
    run_various_embodiments()
    run_strided_chunks()
    run_with_guidance()
    run_loss()
    print("\n[bold green]All XFlowHead demos completed successfully.[/]")


if __name__ == "__main__":
    main()
