"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

from dataclasses import dataclass, field

from flax.training.train_state import TrainState
import grain
import jax
import jax.image
import jax.numpy as jnp
import optax
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
    num_keypoints: int = 0  # 0 = infer from batch
    variant: str = "full"  # quarter | half | full
    mix: Arec = field(default_factory=lambda: Arec.from_name("xarm_dream_100k"))
    sigma: float = 2.0  # TODO desc
    lr: float = 1e-3
    steps: int = 1
    log_every: int = 1
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


def _scaled_kp2d(batch: dict, out_h: int, out_w: int) -> jax.Array:
    in_h, in_w = batch["image"].shape[1:3]
    scale = jnp.array([out_w / in_w, out_h / in_h], dtype=jnp.float32)
    return batch["keypoints_2d"] * scale


def dream_loss_fn(batch: dict, out_dict: dict, sigma: float = 2.0):
    pred = out_dict["pred_heatmaps"]
    _, _, out_h, out_w = pred.shape
    uv = _scaled_kp2d(batch, out_h, out_w)
    vis = batch["keypoints_visible"]
    target = build_heatmaps(uv, vis, image_h=out_h, image_w=out_w, sigma=sigma)
    loss = jnp.mean((pred - target) ** 2)
    metrics = {
        "loss": loss,
        "mse": loss,
        "visible_kp": vis.sum(),
    }
    return loss, metrics


def make_train_step_dream(model, loss_fn):
    @jax.jit
    def train_step(state, batch):
        def _loss(params):
            pred_heatmaps, _ = model.apply({"params": params}, batch["image"])
            out_dict = {"pred_heatmaps": jnp.transpose(pred_heatmaps, (0, 3, 1, 2))}
            return loss_fn(batch, out_dict)

        (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        updates, _ = state.tx.update(grads, state.opt_state, state.params)
        update_info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            **metrics,
        }
        state = state.apply_gradients(grads=grads)
        return state, update_info

    return train_step


def make_eval_step_dream(model, loss_fn):
    @jax.jit
    def eval_step(state, batch):
        pred_heatmaps, _ = model.apply({"params": state.params}, batch["image"])
        out_dict = {"pred_heatmaps": jnp.transpose(pred_heatmaps, (0, 3, 1, 2))}
        loss, metrics = loss_fn(batch, out_dict)
        return {"loss": loss, **metrics}

    return eval_step


def _count_params(params) -> int:
    return sum(x.size for x in jax.tree.leaves(params))


def _print_shapes(shapes):
    table = Table("stage", "shape")
    for name, shape in shapes:
        table.add_row(name, str(shape))
    print(table)


def main(cfg: Config):
    ds = make_dataset(cfg)
    dsit = iter(ds)
    sample = next(dsit)
    batch = prepare_sample(sample)

    print(Rule("DREAM Prepared Sample", style="bold magenta"))
    pprint(spec(batch))

    rng = jax.random.PRNGKey(cfg.seed)
    init_rng = rng
    num_keypoints = cfg.num_keypoints or int(batch["keypoints_2d"].shape[1])
    model = DreamVGG(num_keypoints=num_keypoints, variant=cfg.variant)
    params = model.init(init_rng, batch["image"])["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(cfg.lr),
    )
    loss_fn = lambda batch, out_dict: dream_loss_fn(batch, out_dict, sigma=cfg.sigma)
    train_step = make_train_step_dream(model, loss_fn)
    eval_step = make_eval_step_dream(model, loss_fn)

    pred_heatmaps, shapes = model.apply({"params": state.params}, batch["image"])
    out_dict = {"pred_heatmaps": jnp.transpose(pred_heatmaps, (0, 3, 1, 2))}
    _, init_metrics = loss_fn(batch, out_dict)

    print(Rule("DREAM Forward", style="bold magenta"))
    _print_shapes(shapes)
    print(f"params={_count_params(state.params):,}")
    print(f"pred_heatmaps.shape={out_dict['pred_heatmaps'].shape}")
    print(f"init_loss={float(init_metrics['loss']):.6f}")

    print(Rule("DREAM Train Loop", style="bold magenta"))
    for step in range(cfg.steps):
        batch = prepare_sample(next(dsit))
        state, metrics = train_step(state, batch)
        if step % cfg.log_every == 0:
            print(
                f"step={step} loss={float(metrics['loss']):.6f} "
                f"grad_norm={float(metrics['grad_norm']):.6f} "
                f"visible_kp={int(metrics['visible_kp'])}"
            )

    eval_batch = prepare_sample(next(dsit))
    eval_metrics = eval_step(state, eval_batch)
    print(
        f"eval_loss={float(eval_metrics['loss']):.6f} "
        f"eval_visible_kp={int(eval_metrics['visible_kp'])}"
    )

    if cfg.verbose:
        print(model.tabulate(init_rng, batch["image"], depth=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
