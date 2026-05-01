"""Minimal DREAM VGG encoder-decoder smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path

from flax import struct
from flax.training.train_state import TrainState
import grain
from grain.experimental import ThreadPrefetchIterDataset
import jax
from jax.experimental import multihost_utils
import jax.image
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import numpy as np
import optax
from PIL import Image
from rich import print
from rich.pretty import pprint
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.data.grain.datasets import unpack_record
from crossformer.data.grain.loader import _apply_fd_limit, _grain_mp_worker_init
from crossformer.model.dream import DreamVGG
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import create_optimizer, Timer
import wandb


@dataclass
class DreamVizConfig:
    every: int = 5000


@dataclass
class Optim:
    lr: float = 1.5e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    lr_schedule: str = "constant"  # constant | cosine | rsqrt
    clip_gradient: float | None = 1.0
    acc: int | None = None  # gradient accumulation

    def kwargs(self, steps: int) -> dict:
        learning_rate = self.lr
        if self.lr_schedule != "constant" or self.warmup_steps > 0:
            decay_steps = max(steps, self.warmup_steps + 1)
            learning_rate = {
                "name": self.lr_schedule,
                "init_value": 0.0,
                "peak_value": self.lr,
                "warmup_steps": self.warmup_steps,
                **({"decay_steps": decay_steps} if self.lr_schedule == "cosine" else {}),
            }
        return {
            "learning_rate": learning_rate,
            "weight_decay": self.weight_decay,
            "clip_gradient": self.clip_gradient,
            "grad_accumulation_steps": self.acc,
        }

    def create(self, params, steps: int):
        return create_optimizer(params, **self.kwargs(steps))


@dataclass
class Config:
    """Smoke-test config for DREAM."""

    name: str = "dream"
    seed: int = 0
    steps: int = 1_000_000
    log_every: int = 100
    raw_size: tuple[int, int] = (480, 640)
    net_in_size: tuple[int, int] = (400, 400)
    image_c: int = 3
    num_keypoints: int = 0  # 0 = infer from batch
    variant: str = "full"  # quarter | half | full
    sigma: float = 2.0  # TODO desc
    optim: Optim = default(Optim())
    viz: DreamVizConfig = default(DreamVizConfig())
    wandb: cn.Wandb = default(cn.Wandb(project="bela-dream"))
    verbose: bool = False

    # LOADER
    bs: int = 1
    mix: Arec = default(Arec.from_name("xarm_dream_100k"))
    mp: int = 16
    mp_buf: int = 4  # per worker buffer size
    n_preshard: int = 2  # prefetch sharded data

    coco_prob: float = 0.5
    coco_dir: Path = Path.home() / "bela/datasets/coco/train2014"

    # Checkpointing
    save_dir: Path | None = Path.home().expanduser()
    save_interval: int = 25_000


@struct.dataclass
class DreamCheckpointModel:
    params: dict


@struct.dataclass
class DreamCheckpointState:
    model: DreamCheckpointModel
    step: jax.Array
    opt_state: optax.OptState


def _save_path(cfg: Config) -> str:
    if cfg.save_dir is None:
        raise ValueError("save_dir is None")
    return str((Path(cfg.save_dir).expanduser() / cfg.wandb.project / (cfg.wandb.group or "") / cfg.name).resolve())


def _checkpoint_state(state: TrainState) -> DreamCheckpointState:
    return DreamCheckpointState(
        model=DreamCheckpointModel(params=state.params),
        step=state.step,
        opt_state=state.opt_state,
    )


def _resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = max(target_w, round(src_w * scale))
    new_h = max(target_h, round(src_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def make_coco_dataset(cfg: Config):
    names = sorted(p.name for p in cfg.coco_dir.glob("*.jpg"))
    if not names:
        raise FileNotFoundError(f"no .jpg files under {cfg.coco_dir}")
    net_h, net_w = cfg.net_in_size

    def read_bg(name: str):
        img = Image.open(cfg.coco_dir / name).convert("RGB")
        img = _resize_cover(img, net_w, net_h)
        return {"bg": np.asarray(img, dtype=np.uint8)}

    return grain.MapDataset.source(names).seed(cfg.seed).shuffle().repeat().map(read_bg).to_iter_dataset()


def add_bg(x: dict, rng, prob=0.5):
    if rng.random() < prob:
        x["image"] = np.where(x["mask"][..., None] > 0, x["image"], x["bg"])
    x.pop("bg", None)
    return x


def mix_with_bg(robot_ds, coco_ds, cfg: Config):
    """mix robot dataset with coco background"""

    ds = grain.experimental.ZipIterDataset([robot_ds, coco_ds], strict=False)
    ds = ds.map(lambda pair: pair[0] | pair[1])
    ds = ds.seed(cfg.seed).random_map(partial(add_bg, prob=cfg.coco_prob))
    return ds


def make_dataset(cfg: Config):
    ds = (
        grain.MapDataset.source(cfg.mix.source)
        .seed(42)
        .shuffle()
        .repeat()
        .map(unpack_record)
        .to_iter_dataset(
            grain.ReadOptions(num_threads=32, prefetch_buffer_size=1024)
        )  # iter before batch so that procs do batching and doesnt impede read threads
    )

    ds = ds.map(partial(prepare_sample_np, cfg))
    if cfg.coco_prob:
        coco = make_coco_dataset(cfg)
        ds = mix_with_bg(ds, coco, cfg)
    ds = ds.batch(cfg.bs, drop_remainder=True)

    if cfg.mp > 0:
        lim = _apply_fd_limit(512**2)
        # Workers spawn via multiprocessing and re-import JAX. Without these,
        # each worker claims a CUDA context on GPU:0 and OOMs the parent's model.
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        ds = ds.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_buf),
            worker_init_fn=_grain_mp_worker_init,
        )

    shard_fn = make_shard_fn()
    ds = ds.map(shard_fn)
    ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cfg.n_preshard)
    return ds


def make_shard_fn():
    mesh = Mesh(jax.devices(), axis_names="batch")

    def shard_batch(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    return shard_batch


def _normalize_kp2d(kp2d: jax.Array, h: int, w: int) -> jax.Array:
    return kp2d / jnp.array([w, h], dtype=jnp.float32)


def _denormalize_kp2d(kp2d_norm: jax.Array, h: int, w: int) -> jax.Array:
    return kp2d_norm * jnp.array([w, h], dtype=jnp.float32)


def net_out_size(cfg: Config) -> tuple[int, int]:
    h, w = cfg.net_in_size
    if cfg.variant == "full":
        return h, w
    if cfg.variant == "half":
        return h // 2, w // 2
    if cfg.variant == "quarter":
        return h // 4, w // 4
    raise ValueError(f"unknown variant: {cfg.variant}")


def _normalize_kp2d_np(kp2d: np.ndarray, h: int, w: int) -> np.ndarray:
    return kp2d / np.array([w, h], dtype=np.float32)


def _resize_intrinsics_np(K: np.ndarray, in_h: int, in_w: int, out_h: int, out_w: int) -> np.ndarray:
    sx = out_w / in_w
    sy = out_h / in_h
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return S @ K


def _image_to_float(image: jax.Array) -> jax.Array:
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(jnp.float32) / 255.0
    return image.astype(jnp.float32)


def prepare_sample_np(cfg: Config, sample: dict) -> dict:
    raw_h, raw_w = cfg.raw_size
    net_h, net_w = cfg.net_in_size
    image = np.asarray(sample["image"])
    if tuple(image.shape[:2]) != (raw_h, raw_w):
        raise ValueError(f"expected raw_size={(raw_h, raw_w)} but got {tuple(image.shape[:2])}")
    image = np.asarray(Image.fromarray(image).resize((net_w, net_h), Image.BILINEAR))
    mask = np.asarray(Image.fromarray(sample["mask"]).resize((net_w, net_h), Image.NEAREST))
    joints = np.asarray(sample["state"]["joints"], dtype=np.float32)
    gripper = np.asarray([sample["state"]["gripper"]], dtype=np.float32)
    q = np.concatenate([joints, gripper], axis=-1)
    kp2d_raw = np.asarray(sample["state"]["kp2d"], dtype=np.float32)
    kp2d_norm = _normalize_kp2d_np(kp2d_raw, raw_h, raw_w)
    K = np.asarray(sample["camera"]["intr"]["K"], dtype=np.float32)
    return {
        "image": image,
        "mask": mask,
        "q": q,
        "keypoints_2d_norm": kp2d_norm,
        "keypoints_2d_raw": kp2d_raw,
        "keypoints_visible": np.asarray(sample["info"]["kp_visible"], dtype=bool),
        "K": _resize_intrinsics_np(K, raw_h, raw_w, net_h, net_w),
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


def keypoint_metrics(batch: dict, pred_heatmaps: jax.Array, raw_size: tuple[int, int]):
    pred_uv, pred_conf = extract_keypoints(pred_heatmaps)
    _, _, out_h, out_w = pred_heatmaps.shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *raw_size)
    gt_uv = batch["keypoints_2d_raw"]
    vis = batch["keypoints_visible"]
    err = jnp.linalg.norm(pred_uv - gt_uv, axis=-1)
    vis_f = vis.astype(jnp.float32)
    denom = jnp.maximum(vis_f.sum(), 1.0)
    mean_px = (err * vis_f).sum() / denom
    pck_5 = ((err < 5.0).astype(jnp.float32) * vis_f).sum() / denom
    pck_10 = ((err < 10.0).astype(jnp.float32) * vis_f).sum() / denom
    return {
        "mean_px": mean_px,
        "pck_5": pck_5,
        "pck_10": pck_10,
        "conf_mean": pred_conf.mean(),
    }


def dream_loss_fn(batch: dict, out_dict: dict, sigma: float = 2.0, raw_size: tuple[int, int] = (480, 640)):
    pred = out_dict["pred_heatmaps"]
    _, _, out_h, out_w = pred.shape
    uv = _denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w)
    vis = batch["keypoints_visible"]
    target = build_heatmaps(uv, vis, image_h=out_h, image_w=out_w, sigma=sigma)
    loss = jnp.mean((pred - target) ** 2)
    metrics = {
        "loss": loss,
        "mse": loss,
        "visible_kp": vis.sum(),
        **keypoint_metrics(batch, pred, raw_size=raw_size),
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
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)
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


def _render_heatmap_overlay(image: np.ndarray, hm: np.ndarray):
    import matplotlib.pyplot as plt

    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    image = np.clip(image, 0.0, 1.0)

    hm = np.asarray(hm)
    hm = hm.max(axis=0) if hm.ndim == 3 else hm
    vmax = max(float(hm.max(initial=0.0)), 1e-6)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image)
    ax.imshow(
        hm,
        cmap="magma",
        alpha=0.5,
        vmin=0.0,
        vmax=vmax,
        extent=(0, image.shape[1], image.shape[0], 0),
    )
    ax.set_title("max gt belief map")
    ax.axis("off")
    fig.tight_layout()
    out = wandb.Image(fig)
    plt.close(fig)
    return out


def _render_mask(mask: np.ndarray, idx: int = 0):
    mask = np.asarray(mask[idx])
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255
    elif mask.max(initial=0) <= 1:
        mask = (mask.astype(np.float32) * 255).astype(np.uint8)
    else:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    return wandb.Image(mask)


def maybe_log_viz(cfg: Config, batch: dict, out_dict: dict, step: int):
    if not cfg.wandb.use or cfg.viz.every <= 0 or step % cfg.viz.every != 0:
        return
    pred_uv, pred_conf = extract_keypoints(out_dict["pred_heatmaps"])
    _, _, out_h, out_w = out_dict["pred_heatmaps"].shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    gt_uv = _denormalize_kp2d(batch["keypoints_2d_norm"], *cfg.net_in_size)
    gt_heatmaps = build_heatmaps(
        _denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w),
        batch["keypoints_visible"],
        image_h=out_h,
        image_w=out_w,
        sigma=cfg.sigma,
    )
    log = {
        "viz/predictions": _render_overlay(
            {
                "image": jax.device_get(batch["image"]),
                "keypoints_2d": jax.device_get(gt_uv),
                "keypoints_visible": jax.device_get(batch["keypoints_visible"]),
            },
            jax.device_get(pred_uv),
            jax.device_get(pred_conf),
            jax.device_get(out_dict["pred_heatmaps"]),
        ),
        "viz/gt": _render_heatmap_overlay(jax.device_get(batch["image"][0]), jax.device_get(gt_heatmaps[0])),
        "viz/mask": _render_mask(jax.device_get(batch["mask"])),
    }
    cfg.wandb.log(log, step=step)


def resize_pred_heatmaps(pred_heatmaps: jax.Array, out_h: int, out_w: int) -> jax.Array:
    if tuple(pred_heatmaps.shape[-2:]) == (out_h, out_w):
        return pred_heatmaps
    pred_heatmaps = jnp.transpose(pred_heatmaps, (0, 2, 3, 1))
    pred_heatmaps = jax.image.resize(
        pred_heatmaps,
        (pred_heatmaps.shape[0], out_h, out_w, pred_heatmaps.shape[-1]),
        method="bilinear",
    )
    return jnp.transpose(pred_heatmaps, (0, 3, 1, 2))


def make_train_step_dream(model, loss_fn, out_h: int, out_w: int, lr_fn, param_norm_fn):
    @jax.jit
    def train_step(state, batch):
        image = _image_to_float(batch["image"])

        def _loss(params):
            pred_heatmaps, _ = model.apply({"params": params}, image)
            pred_heatmaps = jnp.transpose(pred_heatmaps, (0, 3, 1, 2))
            pred_heatmaps = resize_pred_heatmaps(pred_heatmaps, out_h, out_w)
            out_dict = {"pred_heatmaps": pred_heatmaps}
            return loss_fn(batch, out_dict)

        (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
        update_info = {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": param_norm_fn(state.params),
            "learning_rate": lr_fn(state.step),
            **metrics,
        }
        state = state.replace(
            step=state.step + 1,
            params=optax.apply_updates(state.params, updates),
            opt_state=opt_state,
        )
        return state, update_info

    return train_step


def make_eval_step_dream(model, loss_fn, out_h: int, out_w: int):
    @jax.jit
    def eval_step(state, batch):
        image = _image_to_float(batch["image"])
        pred_heatmaps, _ = model.apply({"params": state.params}, image)
        pred_heatmaps = jnp.transpose(pred_heatmaps, (0, 3, 1, 2))
        pred_heatmaps = resize_pred_heatmaps(pred_heatmaps, out_h, out_w)
        out_dict = {"pred_heatmaps": pred_heatmaps}
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
    timer = Timer()
    ndev = len(jax.devices())
    if cfg.bs % ndev != 0:
        raise ValueError(f"bs={cfg.bs} must be divisible by device_count={ndev}")
    ds = make_dataset(cfg)
    dsit = iter(ds)
    batch = next(dsit)

    print(Rule("DREAM Prepared Sample", style="bold magenta"))
    pprint(spec(batch))
    run = cfg.wandb.initialize(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    init_rng = rng
    num_keypoints = cfg.num_keypoints or int(batch["keypoints_2d_norm"].shape[1])
    print(f"raw_size={cfg.raw_size} net_in_size={cfg.net_in_size} net_out_size={net_out_size(cfg)}")
    out_h, out_w = net_out_size(cfg)
    model = DreamVGG(num_keypoints=num_keypoints, variant=cfg.variant)
    image = _image_to_float(batch["image"])
    params = model.init(init_rng, image)["params"]
    tx, lr_fn, param_norm_fn = cfg.optim.create(params, steps=cfg.steps)
    print(Rule("optimizer", style="bold magenta"))
    print(f"  config: {cfg.optim.kwargs(cfg.steps)}")
    print(f"  tx: {tx}")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )
    if cfg.save_dir is not None:
        save_dir = _save_path(cfg)
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir — checkpoints disabled[/]")

    loss_fn = lambda batch, out_dict: dream_loss_fn(batch, out_dict, sigma=cfg.sigma, raw_size=cfg.raw_size)
    train_step = make_train_step_dream(
        model, loss_fn, out_h=out_h, out_w=out_w, lr_fn=lr_fn, param_norm_fn=param_norm_fn
    )
    eval_step = make_eval_step_dream(model, loss_fn, out_h=out_h, out_w=out_w)

    pred_heatmaps, shapes = model.apply({"params": state.params}, image)
    pred_heatmaps = jnp.transpose(pred_heatmaps, (0, 3, 1, 2))
    pred_heatmaps = resize_pred_heatmaps(pred_heatmaps, out_h, out_w)
    out_dict = {"pred_heatmaps": pred_heatmaps}
    _, init_metrics = loss_fn(batch, out_dict)

    print(Rule("DREAM Forward", style="bold magenta"))
    _print_shapes(shapes)
    print(f"params={_count_params(state.params):,}")
    print(f"pred_heatmaps.shape={out_dict['pred_heatmaps'].shape}")
    if tuple(out_dict["pred_heatmaps"].shape[-2:]) != (out_h, out_w):
        raise ValueError(
            f"expected net_out_size={(out_h, out_w)} from variant={cfg.variant}, "
            f"got {tuple(out_dict['pred_heatmaps'].shape[-2:])}"
        )
    print(f"init_loss={float(init_metrics['loss']):.6f}")
    maybe_log_viz(cfg, batch, out_dict, step=0)

    print(Rule("DREAM Train Loop", style="bold magenta"))
    for step in tqdm(range(cfg.steps)):
        with timer("data"):
            batch = next(dsit)
        with timer("train_step"):
            state, metrics = train_step(state, batch)

        if step % cfg.log_every == 0:
            with timer("data"):
                eval_batch = next(dsit)
            with timer("eval_step"):
                eval_metrics = eval_step(state, eval_batch)
            times = {f"timer/{k}": v for k, v in timer.get_average_times().items()}
            cfg.wandb.log({"train": metrics, "eval": eval_metrics, **times}, step=step)
            print({**metrics, **eval_metrics, **times})
        if cfg.viz.every > 0 and step % cfg.viz.every == 0:
            pred_heatmaps, _ = model.apply({"params": state.params}, _image_to_float(batch["image"]))
            out_dict = {"pred_heatmaps": resize_pred_heatmaps(jnp.transpose(pred_heatmaps, (0, 3, 1, 2)), out_h, out_w)}
            maybe_log_viz(cfg, batch, out_dict, step=step)
        if cfg.save_interval > 0 and (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback.save(_checkpoint_state(state), step + 1)

    if save_dir is not None:
        save_callback.save(_checkpoint_state(state), cfg.steps)
        save_callback.wait()
    if cfg.verbose:
        print(model.tabulate(init_rng, _image_to_float(batch["image"]), depth=2))
    cfg.wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
