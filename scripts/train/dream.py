"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from rich import print
from rich.rule import Rule
from rich.table import Table
import tyro

from crossformer.model.dream import DreamVGG


@dataclass
class Config:
    """Smoke-test config for DREAM."""

    seed: int = 0
    batch_size: int = 1
    image_h: int = 480
    image_w: int = 640
    image_c: int = 3
    num_keypoints: int = 7
    variant: str = "full"  # quarter | half | full
    verbose: bool = False


def _count_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def _print_shapes(shapes):
    table = Table("stage", "shape")
    for name, shape in shapes:
        table.add_row(name, str(shape))
    print(table)


def main(cfg: Config):
    print(Rule("DREAM VGG Smoke Test", style="bold magenta"))
    print(
        " ".join(
            [
                f"variant={cfg.variant}",
                f"batch={cfg.batch_size}",
                f"image=({cfg.image_h},{cfg.image_w},{cfg.image_c})",
                f"keypoints={cfg.num_keypoints}",
            ]
        )
    )

    rng = jax.random.PRNGKey(cfg.seed)
    init_rng, data_rng = jax.random.split(rng)

    image = jax.random.normal(
        data_rng,
        (cfg.batch_size, cfg.image_h, cfg.image_w, cfg.image_c),
    )
    model = DreamVGG(num_keypoints=cfg.num_keypoints, variant=cfg.variant)

    variables = model.init(init_rng, image)
    heatmaps, shapes = model.apply(variables, image)

    _print_shapes(shapes)
    print(f"params={_count_params(variables['params']):,}")
    print(f"heatmaps.shape={heatmaps.shape}")
    print(f"heatmaps.dtype={heatmaps.dtype}")

    if cfg.verbose:
        print(model.tabulate(init_rng, image, depth=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
