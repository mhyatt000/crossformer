"""Forward pass demo for XFlowHead (Perceiver IO flow-matching action head)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rich import print
from rich.rule import Rule

from crossformer.model.components.base import TokenGroup
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
    """Basic init + forward pass with flow conditioning."""
    print(Rule("XFlowHead: init + forward"))
    B, W, N, E = 4, 2, 8, 512  # batch, window, n_tokens, embed_dim
    H, A = 20, 7  # action_horizon, action_dim

    head = XFlowHead(
        readout_key="readout",
        action_horizon=H,
        action_dim=A,
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
        flow_steps=10,
    )

    rng = jax.random.PRNGKey(0)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)

    # Init (uses zero dummies for time/a_t internally)
    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    # Forward with explicit time and noisy actions
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
    param_count(variables)


def run_various_embodiments():
    """Show flexibility: different action_dim and action_horizon per head."""
    print(Rule("XFlowHead: various embodiments"))
    B, W, N, E = 2, 1, 4, 256
    rng = jax.random.PRNGKey(1)

    configs = [
        {"action_horizon": 4, "action_dim": 7, "label": "7-DOF arm, horizon=4"},
        {"action_horizon": 20, "action_dim": 22, "label": "22-DOF hand, horizon=20"},
        {"action_horizon": 1, "action_dim": 3, "label": "3-DOF gripper, horizon=1"},
        {"action_horizon": 50, "action_dim": 80, "label": "80-DOF humanoid, horizon=50"},
    ]

    for cfg in configs:
        H, A = cfg["action_horizon"], cfg["action_dim"]
        head = XFlowHead(
            readout_key="readout",
            action_horizon=H,
            action_dim=A,
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
        print(f"  {cfg['label']:40s} -> pred {pred.shape}  params={n:,}")


def run_with_guidance():
    """Forward pass with classifier-free guidance tokens."""
    print(Rule("XFlowHead: guidance tokens (CFG)"))
    B, W, N, E = 4, 2, 8, 512
    H, A = 10, 7
    G = 5  # guidance tokens

    head = XFlowHead(
        readout_key="readout",
        action_horizon=H,
        action_dim=A,
        num_query_channels=256,
        num_heads=8,
        num_self_attend_layers=2,
    )

    rng = jax.random.PRNGKey(2)
    init_rng, fwd_rng = jax.random.split(rng)
    outputs = make_transformer_outputs(init_rng, B, W, N, E)

    # Guidance: e.g. language embedding tokens
    guidance_tokens = jax.random.normal(fwd_rng, (B, G, E))
    guidance_mask = jnp.ones((B, G), dtype=jnp.int32)

    variables = head.init({"params": init_rng, "dropout": fwd_rng}, outputs)

    time = jax.random.uniform(fwd_rng, (B, W, 1))
    a_t = jax.random.normal(fwd_rng, (B, W, H, A))

    # Conditioned pass (all guidance visible)
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

    # Unconditioned pass (guidance masked out)
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

    # CFG interpolation
    cfg_scale = 2.0
    pred_cfg = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    show("pred_cfg (scale=2.0)", pred_cfg)
    param_count(variables)


def run_loss():
    """Compute flow matching loss."""
    print(Rule("XFlowHead: loss computation"))
    B, W, N, E = 4, 2, 8, 512
    H, A = 10, 7

    head = XFlowHead(
        readout_key="readout",
        action_horizon=H,
        action_dim=A,
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
    run_with_guidance()
    run_loss()
    print("\n[bold green]All XFlowHead demos completed successfully.[/]")


if __name__ == "__main__":
    main()
