"""Forward pass demo for Perceiver IO encoder/decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from rich import print
from rich.rule import Rule as _Rule

from crossformer.model.components.heads.io import (
    BasicDecoder,
    Perceiver,
    PerceiverEncoder,
    ProjectionDecoder,
)
from crossformer.utils.spec import spec


def Rule(*args, **kwargs):
    """helper. add newline above/below _Rule for better separation in output."""
    print()
    return _Rule(*args, **kwargs)


def show(label, tree):
    """Print spec (shape, dtype) for a pytree or single array."""
    if tree is None:
        print(f"  [dim]{label}[/]: None")
    elif hasattr(tree, "shape"):
        tree = {label: tree}
        print(spec(tree))
    else:
        print(spec(tree))


def param_count(variables):
    n = sum(x.size for x in jax.tree.leaves(variables["params"]))
    print(f"  [bold green]params[/]: {n:,}")


def run_encoder_only():
    """Standalone encoder: cross-attend inputs into latents, self-attend."""
    print(Rule("Encoder Only"))
    batch, seq_len, input_dim = 2, 128, 64
    z_index_dim, z_channels = 32, 256

    encoder = PerceiverEncoder(
        num_self_attends_per_block=2,
        num_blocks=1,
        z_index_dim=z_index_dim,
        num_z_channels=z_channels,
        num_cross_attend_heads=1,
        num_self_attend_heads=4,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
    )

    rng = jax.random.PRNGKey(0)
    inputs = jax.random.normal(rng, (batch, seq_len, input_dim))
    z_dummy = jnp.zeros((batch, z_index_dim, z_channels))

    variables = encoder.init(rng, inputs, z_dummy)
    z_out = encoder.apply(variables, inputs, z_dummy)

    print("[bold]inputs / encoded:[/]")
    show("inputs", inputs)
    show("z_dummy", z_dummy)
    show("z_out", z_out)

    print("[bold]param tree:[/]")
    # print(spec(variables["params"]))
    param_count(variables)


def run_projection_decoder():
    """Perceiver with ProjectionDecoder (classification head)."""
    print(Rule("Perceiver + ProjectionDecoder (classification)"))
    batch, seq_len, input_dim = 2, 128, 64
    z_index_dim, z_channels = 32, 256
    num_classes = 10

    model = Perceiver(
        encoder=PerceiverEncoder(
            num_self_attends_per_block=2,
            num_blocks=1,
            z_index_dim=z_index_dim,
            num_z_channels=z_channels,
            num_cross_attend_heads=1,
            num_self_attend_heads=4,
        ),
        decoder=ProjectionDecoder(num_classes=num_classes),
    )

    rng = jax.random.PRNGKey(42)
    inputs = jax.random.normal(rng, (batch, seq_len, input_dim))

    variables = model.init(rng, inputs)
    output = model.apply(variables, inputs)

    print("[bold]io:[/]")
    show("inputs", inputs)
    show("output", output)
    print(f"  expected: ({batch}, {num_classes})")

    print("[bold]param tree:[/]")
    # print(spec(variables["params"]))
    param_count(variables)


def run_basic_decoder():
    """Perceiver with BasicDecoder (cross-attention output query)."""
    print(Rule("Perceiver + BasicDecoder (dense output)"))
    batch, seq_len, input_dim = 2, 128, 64
    z_index_dim, z_channels = 32, 256
    output_index_dims = (16,)
    output_channels = 80

    model = Perceiver(
        encoder=PerceiverEncoder(
            num_self_attends_per_block=2,
            num_blocks=1,
            z_index_dim=z_index_dim,
            num_z_channels=z_channels,
            num_cross_attend_heads=1,
            num_self_attend_heads=4,
        ),
        decoder=BasicDecoder(
            output_num_channels=output_channels,
            position_encoding_type="trainable",
            output_index_dims=output_index_dims,
            num_z_channels=z_channels,
            num_heads=1,
            position_encoding_kwargs={
                "trainable_position_encoding_kwargs": {"num_channels": z_channels},
            },
        ),
    )

    rng = jax.random.PRNGKey(7)
    inputs = jax.random.normal(rng, (batch, seq_len, input_dim))

    variables = model.init(rng, inputs)
    output = model.apply(variables, inputs)

    print("[bold]io:[/]")
    show("inputs", inputs)
    show("output", output)
    print(f"  expected: ({batch}, {output_index_dims[0]}, {output_channels})")

    print("[bold]param tree:[/]")
    # print(spec(variables["params"]))
    param_count(variables)


def run_with_mask():
    """Forward pass with input masking (variable-length sequences)."""
    print(Rule("Perceiver + BasicDecoder (with input mask)"))
    batch, seq_len, input_dim = 2, 128, 64
    z_index_dim, z_channels = 32, 256
    output_index_dims = (16,)
    output_channels = 80

    model = Perceiver(
        encoder=PerceiverEncoder(
            num_self_attends_per_block=2,
            num_blocks=1,
            z_index_dim=z_index_dim,
            num_z_channels=z_channels,
            num_cross_attend_heads=1,
            num_self_attend_heads=4,
        ),
        decoder=BasicDecoder(
            output_num_channels=output_channels,
            position_encoding_type="trainable",
            output_index_dims=output_index_dims,
            num_z_channels=z_channels,
            num_heads=1,
            position_encoding_kwargs={
                "trainable_position_encoding_kwargs": {"num_channels": z_channels},
            },
        ),
    )

    rng = jax.random.PRNGKey(13)
    inputs = jax.random.normal(rng, (batch, seq_len, input_dim))

    # mask: first sample has 100 valid tokens, second has 64
    input_mask = jnp.zeros((batch, seq_len), dtype=jnp.int32)
    input_mask = input_mask.at[0, :100].set(1)
    input_mask = input_mask.at[1, :64].set(1)

    variables = model.init(rng, inputs, input_mask=input_mask)
    output = model.apply(variables, inputs, input_mask=input_mask)

    print("[bold]io:[/]")
    show("inputs", inputs)
    show("input_mask", input_mask)
    show("output", output)
    print(f"  mask sums: {input_mask.sum(axis=1).tolist()}")

    print("[bold]param tree:[/]")
    # print(spec(variables["params"]))
    param_count(variables)


def main():
    print(Rule("Perceiver IO Forward Pass Demo", style="bold magenta"))

    run_encoder_only()
    run_projection_decoder()
    run_basic_decoder()
    run_with_mask()

    print("\n[bold green]All forward passes completed successfully.[/]")


if __name__ == "__main__":
    main()
