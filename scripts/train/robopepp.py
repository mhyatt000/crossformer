"""RoboPEPP single-file training scaffold.

This file intentionally reuses DREAM data, viz, and pose utilities. RoboPEPP
pieces that need broader repo support are marked TODO rather than hidden behind
half-local abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flax import linen as nn
from flax import struct
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rich import print
from rich.pretty import pprint
from rich.rule import Rule
from rich.table import Table
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset.mix import Arec
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.callbacks.synth_viz import fk_keypoints, solve_pnp
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import Timer
from scripts.train.dream import (
    _count_params,
    _denormalize_kp2d,
    _image_to_float,
    _project_points,
    _rot_err_deg,
    _transform_points,
    ADD_THRESHOLDS_MM,
    build_heatmaps,
    DreamVizConfig,
    extract_keypoints,
    keypoint_metrics,
    make_dataset,
    maybe_log_viz,
    Optim,
)
import wandb


@dataclass
class RoboPeppConfig:
    """RoboPEPP supervised fine-tuning scaffold."""

    name: str = "robopepp"
    seed: int = 0
    steps: int = 1_000_000
    log_every: int = 100

    raw_size: tuple[int, int] = (480, 640)
    net_in_size: tuple[int, int] = (224, 224)
    image_c: int = 3
    num_keypoints: int = 0  # 0 = infer from batch
    q_dim: int = 0  # 0 = infer from batch

    patch: int = 16
    embed_dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    pred_dim: int = 384
    pred_depth: int = 4
    joint_iters: int = 4

    sigma: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    q_weight: float = 1.0
    steps_per_epoch: int = 1000  # TODO derive from finite train set when epochs are first-class.
    pnp_tau: float = 0.03

    optim: Optim = default(Optim())
    viz: DreamVizConfig = default(DreamVizConfig())
    wandb: cn.Wandb = default(cn.Wandb(project="bela-robopepp"))
    verbose: bool = False

    # Loader fields consumed by scripts.train.dream.make_dataset.
    bs: int = 1
    mix: Arec = default(Arec.from_name("xarm_dream_100k"))
    mp: int = 16
    mp_buf: int = 4
    n_preshard: int = 2
    coco_prob: float = 0.5
    coco_dir: Path = Path.home() / "bela/datasets/coco/train2014"

    save_dir: Path | None = Path.home().expanduser()
    save_interval: int = 25_000

    # TODO: pretrain stage needs mask generation around joints, target-encoder
    # EMA params, and a separate checkpoint shape.
    stage: str = "finetune"  # finetune | pretrain placeholder | s2r placeholder


@struct.dataclass
class RoboPeppCheckpointModel:
    params: dict


@struct.dataclass
class RoboPeppCheckpointState:
    model: RoboPeppCheckpointModel
    step: jax.Array
    opt_state: optax.OptState


def _save_path(cfg: RoboPeppConfig) -> str:
    if cfg.save_dir is None:
        raise ValueError("save_dir is None")
    return str((Path(cfg.save_dir).expanduser() / cfg.wandb.project / (cfg.wandb.group or "") / cfg.name).resolve())


def _checkpoint_state(state: TrainState) -> RoboPeppCheckpointState:
    return RoboPeppCheckpointState(
        model=RoboPeppCheckpointModel(params=state.params),
        step=state.step,
        opt_state=state.opt_state,
    )


class TransformerBlock(nn.Module):
    dim: int
    heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(num_heads=self.heads, qkv_features=self.dim, out_features=self.dim)(h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp_dim)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return x + h


class PatchViTEncoder(nn.Module):
    patch: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, image):
        x = nn.Conv(self.dim, (self.patch, self.patch), strides=(self.patch, self.patch), padding="VALID", name="patch")(
            image
        )
        b, gh, gw, c = x.shape
        x = x.reshape(b, gh * gw, c)
        # TODO: initialize learned pos_embed from fixed 2D sin-cos embeddings.
        pos = self.param("pos_embed", nn.initializers.normal(stddev=0.02), (1, gh * gw, c))
        x = x + pos
        for i in range(self.depth):
            x = TransformerBlock(self.dim, self.heads, self.mlp_dim, name=f"block{i}")(x)
        x = nn.LayerNorm(name="norm")(x)
        return x, (gh, gw)


class JepaPredictor(nn.Module):
    """JEPA predictor placeholder.

    TODO: wire visible-token selection, learned mask tokens, and target-encoder
    EMA training. This module exists so params/checkpoints have the intended
    shape once pretraining is added.
    """

    in_dim: int
    pred_dim: int
    depth: int
    heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, tokens):
        x = nn.Dense(self.pred_dim, name="in_proj")(tokens)
        for i in range(self.depth):
            x = TransformerBlock(self.pred_dim, self.heads, self.mlp_dim, name=f"block{i}")(x)
        return nn.Dense(self.in_dim, name="out_proj")(x)


class JointHead(nn.Module):
    q_dim: int
    iters: int = 4
    hidden: int = 512

    @nn.compact
    def __call__(self, tokens):
        z = tokens.mean(axis=1)
        q = jnp.zeros((tokens.shape[0], self.q_dim), dtype=tokens.dtype)
        for _ in range(self.iters):
            h = jnp.concatenate([z, q], axis=-1)
            h = nn.Dense(self.hidden)(h)
            h = nn.gelu(h)
            h = nn.Dense(self.hidden)(h)
            h = nn.gelu(h)
            q = q + nn.Dense(self.q_dim)(h)
        return q


class KeypointHead(nn.Module):
    num_keypoints: int

    @nn.compact
    def __call__(self, tokens, grid: tuple[int, int]):
        gh, gw = grid
        x = tokens.reshape(tokens.shape[0], gh, gw, tokens.shape[-1])
        for i in range(4):
            x = nn.ConvTranspose(256, (4, 4), strides=(2, 2), padding="SAME", name=f"up{i + 1}")(x)
            # TODO: RoboPEPP uses BatchNorm + dropout. GroupNorm keeps this
            # scaffold stateless so it can share DREAM's simple TrainState.
            x = nn.GroupNorm(num_groups=32)(x)
            x = nn.relu(x)
        x = nn.Conv(self.num_keypoints, (1, 1), padding="SAME", name="heatmap")(x)
        return nn.sigmoid(x)


class RoboPepp(nn.Module):
    num_keypoints: int
    q_dim: int
    patch: int = 16
    embed_dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_dim: int = 3072
    pred_dim: int = 384
    pred_depth: int = 4
    joint_iters: int = 4

    @nn.compact
    def __call__(self, image, train: bool = True):
        del train  # TODO: use for dropout once model state includes batch/dropout rngs.
        tokens, grid = PatchViTEncoder(
            patch=self.patch,
            dim=self.embed_dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            name="encoder",
        )(image)
        pred_tokens = JepaPredictor(
            in_dim=self.embed_dim,
            pred_dim=self.pred_dim,
            depth=self.pred_depth,
            heads=max(1, self.heads // 2),
            mlp_dim=self.pred_dim * 4,
            name="predictor",
        )(tokens)
        q_hat = JointHead(self.q_dim, iters=self.joint_iters, name="joint")(pred_tokens)
        heatmaps = KeypointHead(self.num_keypoints, name="keypoint")(pred_tokens, grid)
        heatmaps = jnp.transpose(heatmaps, (0, 3, 1, 2))
        return {"pred_heatmaps": heatmaps, "q_hat": q_hat, "tokens": tokens, "pred_tokens": pred_tokens}


def q_lambda(step: jax.Array, steps_per_epoch: int) -> jax.Array:
    epoch = step // steps_per_epoch
    return jnp.where(epoch < 5, 1e-4, jnp.where(epoch < 10, 1e-2, jnp.where(epoch < 40, 1e-1, 1.0)))


def sigmoid_focal_loss(pred: jax.Array, target: jax.Array, alpha: float = 0.25, gamma: float = 2.0) -> jax.Array:
    pred = jnp.clip(pred, 1e-4, 1.0 - 1e-4)
    bce = -(target * jnp.log(pred) + (1.0 - target) * jnp.log1p(-pred))
    pt = target * pred + (1.0 - target) * (1.0 - pred)
    alpha_t = target * alpha + (1.0 - target) * (1.0 - alpha)
    return alpha_t * ((1.0 - pt) ** gamma) * bce


def robopepp_loss_fn(cfg: RoboPeppConfig, batch: dict, out: dict, step: jax.Array):
    pred = out["pred_heatmaps"]
    _, _, out_h, out_w = pred.shape
    uv = _denormalize_kp2d(batch["keypoints_2d_norm"], out_h, out_w)
    vis = batch["keypoints_visible"]
    target = build_heatmaps(uv, vis, image_h=out_h, image_w=out_w, sigma=cfg.sigma)
    kp_loss_map = sigmoid_focal_loss(pred, target, alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    kp_loss = kp_loss_map.mean()

    q_gt = batch["q"][:, : out["q_hat"].shape[-1]]
    # TODO: use dataset mean/std normalization for q, as in the RoboPEPP paper.
    q_loss = jnp.mean((out["q_hat"] - q_gt) ** 2)
    q_w = cfg.q_weight * q_lambda(step, cfg.steps_per_epoch)
    loss = kp_loss + q_w * q_loss
    metrics = {
        "loss": loss,
        "kp_focal": kp_loss,
        "q_mse": q_loss,
        "q_lambda": q_w,
        "visible_kp": vis.sum(),
        **keypoint_metrics(batch, pred, raw_size=cfg.raw_size),
    }
    return loss, metrics


def _valid_from_conf(conf: np.ndarray, tau: float) -> np.ndarray:
    threshold = tau
    valid = np.isfinite(conf) & (conf > threshold)
    while valid.sum() < 4 and threshold > 0.0:
        threshold -= 0.025
        valid = np.isfinite(conf) & (conf > threshold)
    return valid


def _pose_metrics_one(q_hat, uv_px, conf, K, w2c_gt, tau: float) -> dict:
    joints_rad = np.deg2rad(np.asarray(q_hat[:7], dtype=np.float64))
    pts_3d = fk_keypoints(joints_rad)
    valid = np.isfinite(uv_px).all(axis=-1) & _valid_from_conf(conf, tau)
    out = {"valid_kp": float(valid.sum()), "success": 0.0}
    try:
        w2c_pred = solve_pnp(pts_3d, uv_px, K, valid)
    except Exception:
        return out

    reproj = _project_points(w2c_pred, pts_3d, K)
    reproj_err = np.linalg.norm(reproj[valid] - uv_px[valid], axis=-1).mean()
    pred_cam = _transform_points(w2c_pred, pts_3d)
    gt_cam = _transform_points(w2c_gt, pts_3d)
    add_mm = np.linalg.norm(pred_cam - gt_cam, axis=-1).mean() * 1000.0
    return {
        **out,
        "success": 1.0,
        "reproj_px": float(reproj_err),
        "add_mm": float(add_mm),
        "rot_err_deg": _rot_err_deg(w2c_pred[:3, :3], w2c_gt[:3, :3]),
        "trans_err_mm": float(np.linalg.norm(w2c_pred[:3, 3] - w2c_gt[:3, 3]) * 1000.0),
    }


def pose_metrics(cfg: RoboPeppConfig, batch: dict, out: dict) -> dict:
    pred_uv, conf = extract_keypoints(out["pred_heatmaps"])
    _, _, out_h, out_w = out["pred_heatmaps"].shape
    pred_uv = _denormalize_kp2d(pred_uv / jnp.array([out_w, out_h], dtype=jnp.float32), *cfg.net_in_size)
    batch_np = jax.device_get(batch)
    uv_np = np.asarray(jax.device_get(pred_uv), dtype=np.float64)
    conf_np = np.asarray(jax.device_get(conf), dtype=np.float64)
    q_np = np.asarray(jax.device_get(out["q_hat"]), dtype=np.float64)
    K_np = np.asarray(batch_np["K"], dtype=np.float64)
    w2c_np = np.asarray(batch_np["w2c"], dtype=np.float64)

    rows = [_pose_metrics_one(q_np[i], uv_np[i], conf_np[i], K_np[i], w2c_np[i], cfg.pnp_tau) for i in range(q_np.shape[0])]
    vals = {}
    for key in ("valid_kp", "success", "reproj_px", "add_mm", "rot_err_deg", "trans_err_mm"):
        xs = np.asarray([r[key] for r in rows if key in r], dtype=np.float32)
        vals[key] = float(xs.mean()) if len(xs) else float("nan")
    adds = np.asarray([r["add_mm"] for r in rows if "add_mm" in r], dtype=np.float32)
    if len(adds):
        curve = (adds[:, None] < ADD_THRESHOLDS_MM[None]).mean(axis=0)
        vals["add_auc_100mm"] = float(
            (((curve[1:] + curve[:-1]) * 0.5).sum() * (ADD_THRESHOLDS_MM[1] - ADD_THRESHOLDS_MM[0]))
            / ADD_THRESHOLDS_MM[-1]
        )
    else:
        vals["add_auc_100mm"] = float("nan")
    return vals


def make_train_step(model: RoboPepp, cfg: RoboPeppConfig, lr_fn, param_norm_fn):
    @jax.jit
    def train_step(state, batch):
        image = _image_to_float(batch["image"])

        def _loss(params):
            out = model.apply({"params": params}, image, train=True)
            return robopepp_loss_fn(cfg, batch, out, state.step)

        (loss, metrics), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)
        updates, opt_state = state.tx.update(grads, state.opt_state, state.params)
        state = state.replace(
            step=state.step + 1,
            params=optax.apply_updates(state.params, updates),
            opt_state=opt_state,
        )
        return state, {
            "loss": loss,
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "param_norm": param_norm_fn(state.params),
            "learning_rate": lr_fn(state.step),
            **metrics,
        }

    return train_step


def make_eval_step(model: RoboPepp, cfg: RoboPeppConfig):
    @jax.jit
    def eval_step(state, batch):
        out = model.apply({"params": state.params}, _image_to_float(batch["image"]), train=False)
        _, metrics = robopepp_loss_fn(cfg, batch, out, state.step)
        return metrics

    return eval_step


def _print_shapes(out: dict):
    table = Table("name", "shape")
    for k, v in out.items():
        table.add_row(k, str(v.shape))
    print(table)


def _not_implemented_stage(stage: str):
    raise NotImplementedError(
        f"stage={stage!r} is only sketched in this single-file scaffold. "
        "TODO: add JEPA mask sampling + EMA target params for pretrain, or "
        "differentiable PnP + keypoint stop-gradient plumbing for s2r."
    )


def main(cfg: RoboPeppConfig):
    if cfg.stage != "finetune":
        _not_implemented_stage(cfg.stage)

    timer = Timer()
    ndev = len(jax.devices())
    if cfg.bs % ndev != 0:
        raise ValueError(f"bs={cfg.bs} must be divisible by device_count={ndev}")

    ds = make_dataset(cfg)
    dsit = iter(ds)
    batch = next(dsit)

    print(Rule("RoboPEPP Prepared Sample", style="bold magenta"))
    pprint(spec(batch))
    cfg.wandb.initialize(cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    num_keypoints = cfg.num_keypoints or int(batch["keypoints_2d_norm"].shape[1])
    q_dim = cfg.q_dim or int(batch["q"].shape[-1])
    model = RoboPepp(
        num_keypoints=num_keypoints,
        q_dim=q_dim,
        patch=cfg.patch,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        mlp_dim=cfg.mlp_dim,
        pred_dim=cfg.pred_dim,
        pred_depth=cfg.pred_depth,
        joint_iters=cfg.joint_iters,
    )
    image = _image_to_float(batch["image"])
    params = model.init(rng, image, train=False)["params"]
    tx, lr_fn, param_norm_fn = cfg.optim.create(params, steps=cfg.steps)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    print(Rule("optimizer", style="bold magenta"))
    print(f"  config: {cfg.optim.kwargs(cfg.steps)}")
    print(f"  tx: {tx}")
    if cfg.save_dir is not None:
        save_dir = _save_path(cfg)
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir - checkpoints disabled[/]")

    train_step = make_train_step(model, cfg, lr_fn=lr_fn, param_norm_fn=param_norm_fn)
    eval_step = make_eval_step(model, cfg)

    out = model.apply({"params": state.params}, image, train=False)
    _, init_metrics = robopepp_loss_fn(cfg, batch, out, state.step)
    print(Rule("RoboPEPP Forward", style="bold magenta"))
    _print_shapes(out)
    print(f"params={_count_params(state.params):,}")
    print(f"init_loss={float(init_metrics['loss']):.6f}")
    maybe_log_viz(cfg, batch, out, step=0)

    print(Rule("RoboPEPP Train Loop", style="bold magenta"))
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
            with timer("pose"):
                eval_out = model.apply({"params": state.params}, _image_to_float(eval_batch["image"]), train=False)
                pnp_metrics = pose_metrics(cfg, eval_batch, eval_out)
            times = {f"timer/{k}": v for k, v in timer.get_average_times().items()}
            cfg.wandb.log({"train": metrics, "eval": eval_metrics, "pose": pnp_metrics, **times}, step=step)
            print({**metrics, **eval_metrics, **pnp_metrics, **times})

        if cfg.viz.every > 0 and step % cfg.viz.every == 0:
            out = model.apply({"params": state.params}, _image_to_float(batch["image"]), train=False)
            maybe_log_viz(cfg, batch, out, step=step)

        if cfg.save_interval > 0 and (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback.save(_checkpoint_state(state), step + 1)

    if save_dir is not None:
        save_callback.save(_checkpoint_state(state), cfg.steps)
        save_callback.wait()
    if cfg.verbose:
        print(model.tabulate(rng, _image_to_float(batch["image"]), train=False, depth=2))
    cfg.wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(RoboPeppConfig))
