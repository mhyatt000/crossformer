"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

from dataclasses import dataclass, field

from flax.training.train_state import TrainState
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich import print
from rich.pretty import pprint
from rich.rule import Rule
from rich.table import Table
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.datasets import unpack_record
from crossformer.model.dream import DreamVGG
from crossformer.utils.spec import spec
import wandb


@dataclass
class DreamVizConfig:
    every: int = 5000


@dataclass
class Config:
    """Smoke-test config for DREAM."""

    seed: int = 0
    bs: int = 1
    steps: int = 1_000_000
    log_every: int = 100
    image_h: int = 480
    image_w: int = 640
    image_c: int = 3
    num_keypoints: int = 0  # 0 = infer from batch
    variant: str = "full"  # quarter | half | full
    mix: Arec = field(default_factory=lambda: Arec.from_name("xarm_dream_100k"))
    sigma: float = 2.0  # TODO desc
    lr: float = 1e-3
    vis: DreamVizConfig = default(DreamVizConfig())
    wandb: cn.Wandb = default(cn.Wandb(project="bela-dream"))
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


def _unscale_kp2d(kp2d: jax.Array, image_h: int, image_w: int, out_h: int, out_w: int) -> jax.Array:
    scale = jnp.array([image_w / out_w, image_h / out_h], dtype=jnp.float32)
    return kp2d * scale


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


def _extract_keypoints_one(hm: jax.Array) -> tuple[jax.Array, jax.Array]:
    k, h, w = hm.shape
    flat = hm.reshape(k, h * w)
    idx = jnp.argmax(flat, axis=-1)
    conf = jnp.take_along_axis(flat, idx[:, None], axis=-1)[:, 0]
    ys = idx // w
    xs = idx % w
    uv = jnp.stack([xs, ys], axis=-1).astype(jnp.float32)
    return uv, conf


def extract_keypoints(pred_heatmaps: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jax.vmap(_extract_keypoints_one)(pred_heatmaps)


def _render_overlay(batch: dict, pred_uv: np.ndarray, pred_conf: np.ndarray, pred_heatmaps: np.ndarray, idx: int = 0):
    import matplotlib.pyplot as plt

    image = np.asarray(batch["image"][idx])
    uv_gt = np.asarray(batch["keypoints_2d"][idx])
    vis = np.asarray(batch["keypoints_visible"][idx])
    uv_pred = np.asarray(pred_uv[idx])
    conf = np.asarray(pred_conf[idx])
    hm = np.asarray(pred_heatmaps[idx])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image)
    axes[0].scatter(uv_gt[vis, 0], uv_gt[vis, 1], c="lime", s=20, label="gt")
    axes[0].scatter(uv_pred[:, 0], uv_pred[:, 1], c="red", s=20, label="pred")
    axes[0].set_title("image + keypoints")
    axes[0].legend()
    axes[0].axis("off")

    axes[1].imshow(hm.max(axis=0), cmap="magma")
    axes[1].set_title("max heatmap")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(hm.max(axis=0), cmap="magma", alpha=0.5)
    axes[2].set_title(f"overlay conf={conf.mean():.3f}")
    axes[2].axis("off")

    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def maybe_log_viz(cfg: Config, batch: dict, out_dict: dict, step: int):
    if not cfg.wandb.use or cfg.vis.every <= 0 or step % cfg.vis.every != 0:
        return
    pred_uv, pred_conf = extract_keypoints(out_dict["pred_heatmaps"])
    _, _, out_h, out_w = out_dict["pred_heatmaps"].shape
    pred_uv = _unscale_kp2d(pred_uv, batch["image"].shape[1], batch["image"].shape[2], out_h, out_w)
    log = {
        "viz/predictions": _render_overlay(
            jax.device_get(batch),
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
            jax.device_get(out_dict["pred_heatmaps"]),
        )
    }
    cfg.wandb.log(log, step=step)


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
    run = cfg.wandb.initialize(cfg)

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
    maybe_log_viz(cfg, batch, out_dict, step=0)

    print(Rule("DREAM Train Loop", style="bold magenta"))
    for step in range(cfg.steps):
        batch = prepare_sample(next(dsit))
        state, metrics = train_step(state, batch)
        eval_metrics = None
        if step % cfg.log_every == 0:
            eval_batch = prepare_sample(next(dsit))
            eval_metrics = eval_step(state, eval_batch)
            cfg.wandb.log({"train": metrics, "eval": eval_metrics}, step=step)
            print(
                f"step={step} loss={float(metrics['loss']):.6f} "
                f"eval_loss={float(eval_metrics['loss']):.6f} "
                f"grad_norm={float(metrics['grad_norm']):.6f} "
                f"visible_kp={int(metrics['visible_kp'])}"
            )
        if cfg.vis.every > 0 and step % cfg.vis.every == 0:
            pred_heatmaps, _ = model.apply({"params": state.params}, batch["image"])
            out_dict = {"pred_heatmaps": jnp.transpose(pred_heatmaps, (0, 3, 1, 2))}
            maybe_log_viz(cfg, batch, out_dict, step=step)

    if cfg.verbose:
        print(model.tabulate(init_rng, batch["image"], depth=2))
    cfg.wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
