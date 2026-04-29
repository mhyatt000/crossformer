"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

from dataclasses import dataclass, field

import grain
import jax
import jax.numpy as jnp
from rich import print
from rich.pretty import pprint
from rich.rule import Rule
from rich.table import Table
import tyro

from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.datasets import unpack_record
from crossformer.model.dream import DreamVGG
from crossformer.utils.spec import spec


@dataclass
class Config:
    """Smoke-test config for DREAM."""

    seed: int = 0
    bs: int = 1
    image_h: int = 480
    image_w: int = 640
    image_c: int = 3
    num_keypoints: int = 7
    variant: str = "full"  # quarter | half | full
    mix: Arec = field(default_factory=lambda: Arec.from_name("xarm_dream_100k"))
    sigma: float = 2.0  # TODO desc
    verbose: bool = False


def make_dataset(cfg: Config):
    ds = (
        grain.MapDataset.source(cfg.mix.source)
        .seed(42)
        .shuffle()
        .repeat()
        .map(unpack_record)
        .to_iter_dataset()
        .batch(cfg.bs, drop_remainder=True)
    )
    return ds


def prepare_sample(sample: dict) -> dict:
    image = jnp.asarray(sample["image"], dtype=jnp.float32) / 255.0
    joints = jnp.asarray(sample["state"]["joints"], dtype=jnp.float32)
    gripper = jnp.asarray(sample["state"]["gripper"], dtype=jnp.float32)[..., None]
    q = jnp.concatenate([joints, gripper], axis=-1)
    return {
        "image": image,
        "q": q,
        "keypoints_2d": jnp.asarray(sample["state"]["kp2d"], dtype=jnp.float32),
        "keypoints_visible": jnp.asarray(sample["info"]["kp_visible"], dtype=jnp.bool_),
        "K": jnp.asarray(sample["camera"]["intr"]["K"], dtype=jnp.float32),
    }


def _build_heatmaps_one(
    kp2d: jax.Array,
    visible: jax.Array,
    image_h: int,
    image_w: int,
    sigma: float = 2.0,
) -> jax.Array:
    ys = jnp.arange(image_h, dtype=jnp.float32)[:, None]
    xs = jnp.arange(image_w, dtype=jnp.float32)[None, :]
    u = kp2d[:, 0][:, None, None]
    v = kp2d[:, 1][:, None, None]
    dist2 = (xs - u) ** 2 + (ys - v) ** 2
    heatmaps = jnp.exp(-dist2 / (2.0 * sigma**2))
    mask = visible[:, None, None]
    return jnp.where(mask, heatmaps, jnp.zeros_like(heatmaps))


def build_heatmaps(
    kp2d: jax.Array,
    visible: jax.Array,
    image_h: int,
    image_w: int,
    sigma: float = 2.0,
) -> jax.Array:
    return jax.vmap(lambda uv, vis: _build_heatmaps_one(uv, vis, image_h=image_h, image_w=image_w, sigma=sigma))(
        kp2d, visible
    )


def _count_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def _print_shapes(shapes):
    table = Table("stage", "shape")
    for name, shape in shapes:
        table.add_row(name, str(shape))
    print(table)


def main(cfg: Config):
    print(cfg)

    ds = make_dataset(cfg)
    dsit = iter(ds)
    sample = next(dsit)

    pprint(spec(sample))

    batch = prepare_sample(sample)
    heatmaps = build_heatmaps(
        batch["keypoints_2d"],
        batch["keypoints_visible"],
        image_h=batch["image"].shape[1],
        image_w=batch["image"].shape[2],
        sigma=cfg.sigma,
    )

    print(Rule("DREAM Prepared Sample", style="bold magenta"))
    pprint(spec(batch))
    pprint(spec({"heatmaps": heatmaps}))

    print(Rule("DREAM VGG Smoke Test", style="bold magenta"))

    rng = jax.random.PRNGKey(cfg.seed)
    init_rng, data_rng = jax.random.split(rng)

    image = jax.random.normal(
        data_rng,
        (cfg.bs, cfg.image_h, cfg.image_w, cfg.image_c),
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
