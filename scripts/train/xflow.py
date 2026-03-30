"""End-to-end XFlowHead training: CrossFormerModel + real data.

Bundled action format: uses act.base / act.id from the grain embody
pipeline instead of per-head action extraction.

Usage:
    uv run scripts/train/xflow.py
    uv run scripts/train/xflow.py --steps 500 --lr 3e-4
    uv run scripts/train/xflow.py --mix xgym_sweep --batch-size 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import re

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from rich import print
from rich.rule import Rule
from rich.table import Table
import tensorflow as tf
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.data.grain.embody import decode_embody_name
from crossformer.embody import DOF
from crossformer.model.components.heads.xflow import XFlowHead
from crossformer.model.components.tokenizers import ImageTokenizer, LowdimObsTokenizer
from crossformer.model.components.transformer import common_transformer_sizes
from crossformer.model.components.vit_encoders import ResNet26FILM, SmallStem
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.train_step import lookup_guide, make_train_step
from crossformer.utils.callbacks.rast import RastCallback
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.callbacks.val_mse import ValMSECallback
from crossformer.utils.callbacks.viz import ActionBatchDenormalizer, ChunkVizCallback, HistVizCallback, VizCallback
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.train_utils import create_optimizer, Timer, TrainState
import wandb

# -- config -------------------------------------------------------------------


@dataclass
class Config:
    """XFlowHead end-to-end training config."""

    name: str = ""
    steps: int = 1_000_000  # training steps (1 for debug)
    lr: float = 1e-3  # learning rate
    log_every: int = 100  # log interval
    batch_size: int = 256  # global batch size
    mix: str = "xgym_sweep"  # dataset mix name
    horizon: int = 20  # action horizon from data pipeline
    transformer_size: str = "dummy"  # transformer size preset
    obs_keys: tuple[str, ...] = ("proprio_.*",)  # lowdim obs keys to tokenize

    # Vision backbone
    use_vision: bool = True  # enable image tokenizer
    vision_encoder: str = "small_stem"  # small_stem | resnet26
    image_keys: tuple[str, ...] = ("image_primary", "image_side", "image_left_wrist")  # image obs keys to tokenize

    # XFlowHead sizing
    head_channels: int = 256  # num_query_channels
    head_depth: int = 2  # num_self_attend_layers
    head_heads: int = 8  # num_heads
    flow_steps: int = 50  # high flow steps for now

    # Optimizer
    weight_decay: float = 1e-4  # adamw weight decay
    warmup_steps: int = 0  # lr warmup steps (0 = no warmup)
    lr_schedule: str = "constant"  # constant | cosine | rsqrt
    clip_gradient: float | None = 1.0  # global gradient clipping (None to disable)
    frozen_keys: tuple[str, ...] = ()  # fnmatch patterns for frozen params

    # Token guidance
    use_guidance: bool = True  # enable guidance tokens
    guidance_prob: float = 0.5  # prob of using guidance each step (0-1, for dropout sweep)
    compress_guidance: bool = False  # compress via perceiver latents
    num_guidance_latents: int = 4  # latent count when compress=True
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")  # dot-paths into batch for guidance signal

    # Checkpointing
    save_dir: str | None = Path().home().expanduser()  # checkpoint root dir (None to disable)
    save_interval: int = 25_000  # save every N steps

    mp: int = 8  # grain multiproc (for data loading)
    hist_every: int = 500  # histogram log interval
    viz_every: int = 500  # trajectory video log interval (0 disables)
    viz_fps: int = 12  # logged trajectory video fps
    val_mse_every: int = 500  # fixed-batch action eval interval (0 disables)
    val_mse_print_sample: bool = True  # print one denormalized sample per eval
    val_mse_sample_idx: int = 0  # sample to print from fixed eval batch

    # Robot rasterisation
    rast_urdf: Path | None = Path("xarm7_standalone.urdf")  # URDF for rast callback (None disables)
    rast_mesh_dir: Path | None = Path("assets")  # mesh dir for URDF
    rast_cams: tuple[Path, ...] = (
        Path("data/cam/over/HT.npz"),
        Path("data/cam/side/HT.npz"),
        Path("data/cam/low/HT.npz"),
    )
    rast_width: int = 256
    rast_height: int = 256

    wandb: cn.Wandb = field(default_factory=cn.Wandb)


# -- helpers ------------------------------------------------------------------


VISION_ENCODERS = {
    "small_stem": SmallStem,
    "resnet26": ResNet26FILM,
}
JOINT_NAMES = tuple(f"j{i}" for i in range(7))
JOINT_IDS = tuple(DOF[name] for name in JOINT_NAMES)
JOINT_ID_TO_IDX = {dof_id: i for i, dof_id in enumerate(JOINT_IDS)}


def make_model_config(cfg, max_h, max_a, max_w, guide_dim=None):
    """Build CrossFormerModel config with XFlowHead."""
    token_dim, transformer_kwargs = common_transformer_sizes(cfg.transformer_size)
    readout_name = "xflow"
    readout_key = f"readout_{readout_name}"

    obs_tokenizers = {
        "lowdim": ModuleSpec.create(
            LowdimObsTokenizer,
            obs_keys=list(cfg.obs_keys),
            dropout_rate=0.2,
        ),
    }
    if cfg.use_vision:
        encoder_cls = VISION_ENCODERS.get(cfg.vision_encoder)
        if encoder_cls is None:
            raise ValueError(f"Unknown vision_encoder={cfg.vision_encoder!r}, choose from {list(VISION_ENCODERS)}")
        obs_tokenizers["image"] = ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=list(cfg.image_keys),
            encoder=ModuleSpec.create(encoder_cls),
        )

    return {
        "model": {
            "observation_tokenizers": obs_tokenizers,
            "task_tokenizers": {},
            "heads": {
                "xflow": ModuleSpec.create(
                    XFlowHead,
                    readout_key=readout_key,
                    max_dofs=max_a,
                    max_horizon=max_h,
                    num_query_channels=cfg.head_channels,
                    num_heads=cfg.head_heads,
                    num_self_attend_layers=cfg.head_depth,
                    flow_steps=cfg.flow_steps,
                    use_guidance=cfg.use_guidance,
                    guidance_embed_dim=token_dim,
                    guidance_input_dim=guide_dim,
                    compress_guidance=cfg.compress_guidance,
                    num_guidance_latents=cfg.num_guidance_latents,
                ),
            },
            "readouts": {readout_name: 4},
            "token_embedding_size": token_dim,
            "transformer_kwargs": transformer_kwargs,
            "max_horizon": max_w,
        },
        "text_processor": None,
    }


def normalize_obs(obs, obs_keys):
    """Flatten selected lowdim inputs to (B, W, D)."""
    out = dict(obs)
    for key in obs_keys:
        x = out[key]
        if x.ndim == 2:
            out[key] = x[..., None]
        elif x.ndim > 3:
            out[key] = x.reshape(*x.shape[:2], -1)
    return out


def resolve_obs_keys(obs, patterns):
    """Resolve regex patterns against real observation keys."""
    keys = []
    for pat in patterns:
        keys.extend(k for k in sorted(obs) if k not in keys and re.fullmatch(pat, k))
    if not keys:
        raise ValueError(f"No observation keys matched {patterns}. available={tuple(sorted(obs))}")
    return tuple(keys)


def extract_bundled_actions(batch, max_h):
    """Extract bundled actions from grain embody pipeline.

    Returns:
        actions: (B, W, H, max_a) — W=1 added if missing.
        dof_ids: (B, max_a) — from act.id.
        chunk_steps: (B, H) — just arange(H) for now.
            NOTE: chunk_steps padding disabled — all positions are valid.
    """
    actions = batch["act"]["base"]  # (B, H, max_a)
    if actions.ndim == 3:
        actions = actions[:, None, :, :]  # (B, 1, H, max_a)
    B = actions.shape[0]
    H = actions.shape[2]

    dof_ids = batch["act"]["id"]  # (B, max_a)
    # NOTE: no chunk_steps padding — just dense arange(H)
    chunk_steps = jnp.tile(jnp.arange(H, dtype=jnp.float32)[None], (B, 1))

    return actions, dof_ids, chunk_steps


def adapt_viz_batch(act, flow):
    """Map bundled actions to canonical j0..j6 order for VizCallback."""
    base = np.asarray(act["base"], dtype=np.float32)
    dof_ids = np.asarray(act["id"])
    flow = np.asarray(flow, dtype=np.float32)
    if base.ndim == 3:
        base = base[:, None, :, :]
    if base.ndim != 4:
        raise ValueError(f"Expected act.base ndim 3 or 4, got {base.shape}")
    if flow.ndim != 5:
        raise ValueError(f"Expected flow ndim 5, got {flow.shape}")
    if dof_ids.ndim != 2:
        raise ValueError(f"Expected act.id ndim 2, got {dof_ids.shape}")
    if base.shape[0] != dof_ids.shape[0] or flow.shape[1] != dof_ids.shape[0]:
        raise ValueError(f"Batch mismatch: base={base.shape} flow={flow.shape} dof_ids={dof_ids.shape}")

    keep = []
    base_joint = np.zeros((*base.shape[:-1], len(JOINT_IDS)), dtype=np.float32)
    flow_joint = np.zeros((*flow.shape[:-1], len(JOINT_IDS)), dtype=np.float32)
    for b, row in enumerate(dof_ids):
        has_joint = False
        for src, dof_id in enumerate(row):
            dst = JOINT_ID_TO_IDX.get(int(dof_id))
            if dst is None:
                continue
            has_joint = True
            base_joint[b, ..., dst] = base[b, ..., src]
            flow_joint[:, b, ..., dst] = flow[:, b, ..., src]
        if has_joint:
            keep.append(b)

    if not keep:
        return None, None
    keep = np.asarray(keep, dtype=np.int32)
    return {
        "act": {"base": base_joint[keep]},
        "predict": flow_joint[:, keep],
    }, keep


def denorm_joints(arr: np.ndarray, denorm: ActionBatchDenormalizer, ds_name: str) -> np.ndarray:
    """Denormalize a (..., 7) array of j0..j6 joint values to radians."""
    shape = arr.shape
    means, stds = np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)
    stats = denorm._action_stats(ds_name)
    for j in range(7):
        stat = denorm._dof_array_stats(stats, f"j{j}")
        if stat is not None:
            means[j] = float(np.asarray(stat.mean).reshape(-1)[0])
            stds[j] = float(np.asarray(stat.std).reshape(-1)[0])
    return arr * stds + means


def per_embodiment_metrics(batch, update_info):
    """Compute per-embodiment loss from sample_mse and act.embody.

    Returns dict like {"embodiment/single": mse, "embodiment/dual_arm": mse, ...}.
    """
    embody_arr = np.array(batch["act"]["embody"])  # (B, 32) uint8
    sample_mse = np.array(update_info["sample_mse"])  # (B,)
    names = [decode_embody_name(embody_arr[i]) for i in range(embody_arr.shape[0])]
    groups: dict[str, list[float]] = {}
    for name, mse in zip(names, sample_mse):
        groups.setdefault(name, []).append(float(mse))
    return {f"embodiment/{k}": sum(v) / len(v) for k, v in groups.items()}


def shard_batch(batch, mesh):
    """Shard a host-local batch across the data axis."""
    return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))


# -- main ---------------------------------------------------------------------


def main(cfg: Config):
    tf.config.set_visible_devices([], "GPU")
    initialize_compilation_cache()
    devices = jax.devices()
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    print(Rule("XFlowHead + CrossFormerModel: bundled actions", style="bold magenta"))
    print(f"  backend={jax.default_backend()} devices={len(devices)}")
    if cfg.batch_size % len(devices) != 0:
        raise ValueError(f"batch_size={cfg.batch_size} must be divisible by devices={len(devices)}")

    max_h = cfg.horizon
    run = cfg.wandb.initialize(cfg)

    # Load data
    print(Rule("loading data"))
    from crossformer.cn.dataset import DataSourceE
    from crossformer.cn.dataset.dataset import Loader
    from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory

    _apply_fd_limit(512**2)
    train_cfg = cn.Train(
        data=cn.Dataset(
            mix=DataSourceE[cfg.mix],
            loader=Loader(use_grain=True, global_batch_size=cfg.batch_size),
        ),
        seed=42,
        verbosity=0,
    )
    dataset = GrainDataFactory(mp=cfg.mp).make(train_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=True)
    dsit = iter(dataset.dataset)
    example_batch = next(dsit)
    val_dataset = GrainDataFactory(mp=cfg.mp).make(train_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=False)
    val_batch = next(iter(val_dataset.dataset))
    print(spec(example_batch))
    obs_keys = resolve_obs_keys(example_batch["observation"], cfg.obs_keys)
    print(f"  obs_keys: {obs_keys}")
    val_batch = dict(val_batch)
    val_batch["observation"] = normalize_obs(val_batch["observation"], obs_keys)

    guide_example = None
    if cfg.use_guidance:
        guide_example = lookup_guide(example_batch, cfg.guide_keys)

    # Get max_a from the bundled action shape
    max_a = example_batch["act"]["id"].shape[-1]
    print(f"  max_h={max_h}  max_a={max_a}")
    print(f"  act.base shape: {example_batch['act']['base'].shape}")
    print(f"  act.id   shape: {example_batch['act']['id'].shape}")

    # Build model
    print(Rule("building CrossFormerModel"))
    max_w = example_batch["observation"]["timestep_pad_mask"].shape[1]
    example_obs = normalize_obs(example_batch["observation"], obs_keys)
    config = make_model_config(cfg, max_h, max_a, max_w, None if guide_example is None else guide_example.shape[-1])
    config["model"]["observation_tokenizers"]["lowdim"] = ModuleSpec.create(
        LowdimObsTokenizer,
        obs_keys=[f"^{re.escape(k)}$" for k in obs_keys],
    )
    init_obs = dict(example_obs)
    if cfg.use_vision:
        init_obs |= {
            k: v for k, v in example_batch["observation"].items() if any(re.fullmatch(pat, k) for pat in cfg.image_keys)
        }
        print(f"  image keys in init_obs: {[k for k in init_obs if 'image' in k or 'depth' in k]}")
    init_batch = {
        "observation": init_obs,
        "task": example_batch.get("task", {"pad_mask_dict": {}}),
    }
    wandb.config.update(
        {
            "example_batch_spec": spec(example_batch),
            "obs_keys": obs_keys,
            "max_a": max_a,
            "model_config": config,
        },
        allow_val_change=True,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, train_rng, pred_rng = jax.random.split(rng, 3)

    model = CrossFormerModel.from_config(
        config,
        init_batch,
        text_processor=None,
        verbose=False,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    model = model.replace(
        params=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.params),
        example_batch=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.example_batch),
    )
    n_params = sum(x.size for x in jax.tree.leaves(model.params))
    print(f"  params: {n_params:,}")
    print(f"  heads: {list(model.module.heads.keys())}")
    if cfg.frozen_keys:
        print(f"  frozen_keys: {list(cfg.frozen_keys)}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # Guidance config sanity check
    if cfg.use_guidance:
        print(Rule("guidance encoder"))
        print(f"  guide_keys={cfg.guide_keys} shape={guide_example.shape}")

    # Optimizer + state
    lr_kwargs: dict = {
        "learning_rate": (
            {
                "name": cfg.lr_schedule,
                "init_value": 0.0,
                "peak_value": cfg.lr,
                "warmup_steps": cfg.warmup_steps,
                **({"decay_steps": cfg.steps} if cfg.lr_schedule == "cosine" else {}),
            }
            if cfg.warmup_steps > 0
            else cfg.lr
        ),
        "weight_decay": cfg.weight_decay,
    }
    if cfg.clip_gradient is not None:
        lr_kwargs["clip_gradient"] = cfg.clip_gradient
    if cfg.frozen_keys:
        lr_kwargs["frozen_keys"] = list(cfg.frozen_keys)
    tx, lr_callable, param_norm_callable = create_optimizer(model.params, **lr_kwargs)
    state = TrainState.create(model=model, tx=tx, rng=train_rng)
    train_step = make_train_step(model.module, lr_callable, param_norm_callable)

    # Checkpointing
    if cfg.save_dir is not None:
        save_dir = tf.io.gfile.join(
            cfg.save_dir,
            cfg.wandb.project,
            cfg.wandb.group or "",
            cfg.name,
        )
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir — checkpoints disabled[/]")

    hist_cb = HistVizCallback(stats=dataset.dataset_statistics)
    chunk_cb = ChunkVizCallback(stats=dataset.dataset_statistics)
    viz_cb = VizCallback(flow_key=("predict",), base_key=("act", "base"), fps=cfg.viz_fps)
    rast_cb = None
    if cfg.rast_urdf is not None:
        rast_cb = RastCallback(
            urdf=cfg.rast_urdf,
            cams=list(cfg.rast_cams) if cfg.rast_cams else None,
            mesh_dir=cfg.rast_mesh_dir,
            width=cfg.rast_width,
            height=cfg.rast_height,
        )
    val_mse_cb = ValMSECallback(
        stats=dataset.dataset_statistics,
        guide_keys=cfg.guide_keys,
        sample_idx=cfg.val_mse_sample_idx,
        print_sample=cfg.val_mse_print_sample,
    )

    # Train
    print(Rule("training"))
    table = Table(title="training")
    table.add_column("step", justify="right", style="cyan")
    table.add_column("loss", justify="right")
    table.add_column("|grad|", justify="right", style="dim")

    losses = []
    timer = Timer()
    guide_rng = np.random.default_rng(42)
    for step in tqdm(range(cfg.steps)):
        timer.tick("total")
        with timer("dataset"):
            batch = next(dsit)
            obs = normalize_obs(batch["observation"], obs_keys)
            task = batch.get("task", {"pad_mask_dict": {}})
            pad_mask = obs["timestep_pad_mask"]

            guide_input = None
            if cfg.use_guidance:
                guide_input = lookup_guide(batch, cfg.guide_keys)
                # Guidance dropout: skip with (1 - guidance_prob) probability
                if cfg.guidance_prob < 1.0 and guide_rng.random() > cfg.guidance_prob:
                    guide_input = None

            actions, dof_ids, chunk_steps = extract_bundled_actions(batch, max_h)

        with timer("train"):
            state, update_info = train_step(
                state,
                obs,
                task,
                pad_mask,
                actions,
                dof_ids,
                chunk_steps,
                guide_input=guide_input,
            )
        timer.tock("total")
        update_info = jax.device_get(update_info)
        total_loss = float(update_info["loss"])
        losses.append(total_loss)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            embody_metrics = {}
            if "embody" in batch.get("act", {}):
                embody_metrics = per_embodiment_metrics(batch, update_info)
            print(f"\n[bold]step={step} loss={total_loss}:[/]")
            if embody_metrics:
                for k, v in sorted(embody_metrics.items()):
                    print(f"  {k}: {v:.4f}")
            row = [str(step), f"{total_loss:.4f}"]
            row.append(f"{float(update_info['grad_norm']):.4f}")
            table.add_row(*row)
            hist_metrics = {}
            need_pred = (cfg.hist_every > 0 and (step % cfg.hist_every == 0 or step == cfg.steps - 1)) or (
                cfg.viz_every > 0 and (step % cfg.viz_every == 0 or step == cfg.steps - 1)
            )
            if need_pred:
                bound = model.module.bind({"params": state.model.params})
                transformer_outputs = bound.crossformer_transformer(
                    obs,
                    task,
                    pad_mask,
                    train=False,
                )
                pred = bound.heads["xflow"].predict_action(
                    transformer_outputs,
                    rng=pred_rng,
                    dof_ids=dof_ids,
                    chunk_steps=chunk_steps,
                    train=False,
                    guide_input=guide_input,
                )
                pred_np = jax.device_get(pred)
                if cfg.hist_every > 0 and (step % cfg.hist_every == 0 or step == cfg.steps - 1):
                    pred_flat = pred_np.reshape(pred_np.shape[0], pred_np.shape[1], -1)
                    hist_metrics = hist_cb(batch, {"predict": pred_flat})
                    chunk_imgs = chunk_cb(batch, {"predict": pred_flat})
                    for k, v in chunk_imgs.items():
                        hist_metrics[f"action_chunks/{k}"] = v
                if cfg.viz_every > 0 and (step % cfg.viz_every == 0 or step == cfg.steps - 1):
                    pred_flow = bound.heads["xflow"].predict_action(
                        transformer_outputs,
                        rng=pred_rng,
                        dof_ids=dof_ids,
                        chunk_steps=chunk_steps,
                        train=False,
                        guide_input=guide_input,
                        accumulate=True,
                    )
                    pred_flow_np = jax.device_get(pred_flow)
                    viz_batch, viz_keep = adapt_viz_batch(batch["act"], pred_flow_np)
                    if viz_batch is not None:
                        frames = viz_cb(viz_batch)
                        hist_metrics["flow_pca/video"] = wandb.Video(
                            np.moveaxis(frames, -1, 1),
                            fps=cfg.viz_fps,
                        )
                        if rast_cb is not None:
                            ds_names = chunk_cb.denorm.decode_dataset_names(
                                jax.device_get(batch["info"]["dataset_name"]),
                            )
                            ds0 = ds_names[int(viz_keep[0])]
                            # sample 0, final denoise, all H chunk steps
                            chunk = viz_batch["predict"][-1, 0]  # (W, H, 7)
                            if chunk.ndim == 3:
                                chunk = chunk[0]  # (H, 7) — drop W=1
                            chunk = denorm_joints(chunk, chunk_cb.denorm, ds0)
                            traj_frames = rast_cb.render_trajectory(chunk)
                            for ci, rf in enumerate(traj_frames):
                                hist_metrics[f"rast/cam_{ci}"] = wandb.Image(rf)
            cfg.wandb.log(
                {
                    "training": update_info,
                    **hist_metrics,
                    "timer": timer.get_average_times(),
                    "guidance_active": guide_input is not None,
                    **embody_metrics,
                },
                step=step,
            )

        val_metrics = val_mse_cb.every(
            model,
            state.model.params,
            val_batch,
            step,
            cfg.val_mse_every,
            pred_rng,
            cfg.use_guidance,
        )
        if val_metrics is not None:
            cfg.wandb.log(val_metrics, step=step)

        if (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback(state, step + 1)

    # Final checkpoint
    if save_dir is not None:
        save_callback(state, cfg.steps)
        save_callback.wait()

    print(table)

    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    ratio = last / first if first > 0 else float("inf")
    print(f"\nloss: {first:.4f} -> {last:.4f}  ({ratio:.2%} of initial)")
    cfg.wandb.log({"summary": {"loss_first": first, "loss_last": last, "loss_ratio": ratio}}, step=cfg.steps - 1)

    if ratio < 0.5:
        print("[bold green]loss decreased — training works[/]")
    else:
        print("[bold yellow]loss did not decrease much — check lr or architecture[/]")

    print("\n[bold green]done.[/]")
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
