"""End-to-end XFlowHead training: CrossFormerModel + real data.

Bundled action format: uses act.base / act.id from the grain embody
pipeline instead of per-head action extraction.

Usage:
    uv run scripts/train/xflow.py
    uv run scripts/train/xflow.py --steps 500 --lr 3e-4
    uv run scripts/train/xflow.py --mix xgym_sweep --batch-size 4
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

from flax.core import unfreeze
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from rich import print
from rich.rule import Rule
from tqdm import tqdm
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.cn.dataset import DataSourceE
from crossformer.cn.dataset.dataset import Loader
from crossformer.cn.model_factory import Vision
from crossformer.data.grain.embody import decode_embody_name
from crossformer.model.components.heads.loss_terms import load_loss_weights
from crossformer.model.components.multiview import load_tips_params
from crossformer.model.bela import BELAModel
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.run.train_step import lookup_guide, make_train_step
from crossformer.run.xflow_eval import EvalLoop
from crossformer.utils.callbacks.base import extract_bundled_actions, flatten_obs
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer
from crossformer.utils.callbacks.hist import ChunkCallback, HistCallback
from crossformer.utils.callbacks.kp3dc_viz import Kp3dcVizCallback
from crossformer.utils.callbacks.rast import RastCallback
from crossformer.utils.callbacks.save import SaveCallback
from crossformer.utils.callbacks.synth_viz import SynthVizCallback
from crossformer.utils.callbacks.val_mse import ValMSECallback
from crossformer.utils.callbacks.viz import FlowPCACallback
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import spec
from crossformer.utils.train_utils import create_optimizer, Timer, TrainState
import wandb

# -- config -------------------------------------------------------------------


@dataclass
class Config:
    """XFlowHead end-to-end training config."""

    name: str = ""
    steps: int = 1_000_000  # training steps (1 for debug)
    lr: float = 1e-4  # learning rate
    log_every: int = 100  # log interval
    batch_size: int = 256  # global batch size
    eval_batch_size: int = 64  # eval loader batch size; keep modest so large train batches still boot
    mix: str = "xgym_sweep"  # dataset mix name
    horizon: int = 20  # action horizon from data pipeline
    verbose: bool = False  # print model tabulation during init
    model: cn.ModelFactory = default(
        cn.ModelFactory(
            size=cn.Size.DETR,
            window=20,
            image_keys=(),
            proprio_keys=(),
            vision=Vision(stacked=True, stacked_encoder="tips", tips_variant="tips_v2_b14", stacked_freeze=True),
        )
    )
    debug: bool = False  # debug mode with smaller model and dataset; overrides some other settings

    # Optimizer
    weight_decay: float = 1e-4  # adamw weight decay
    warmup_steps: int = 2000  # lr warmup steps (0 = no warmup, falls back to constant lr)
    lr_schedule: str = "cosine"  # constant | cosine | rsqrt
    clip_gradient: float | None = 1.0  # global gradient clipping (None to disable)
    frozen_keys: tuple[str, ...] = ()  # fnmatch patterns for frozen params
    loss_weights: str | None = "config/loss-weights.yaml"  # per-DOF flow-loss weight yaml (None to disable)
    subtree_norms: bool = False  # log grad/update norms per top-level param subtree (debug)

    # Token guidance
    use_guidance: bool = False  # enable guidance tokens
    guidance_drop_prob: float = 0.5  # prob of dropping guidance each step
    compress_guidance: bool = False  # compress via perceiver latents
    num_guidance_latents: int = 4  # latent count when compress=True
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")  # dot-paths into batch for guidance signal

    # Checkpointing
    save_dir: str | None = str(Path().home().expanduser())  # checkpoint root dir (None to disable)
    save_interval: int = 25_000  # save every N steps

    train_loader: Loader = default(Loader(use_grain=True))
    batches: int | None = None  # train on only the first n batches, cycled forever (overfit debugging)
    mp: int = 8  # grain multiproc (for data loading)
    rotate: bool = False  # apply augmax.Rotate((-15, 15), p=0.3) in grain pipeline
    resize: tuple[int, int] | None = (64, 64)  # final image size; None disables all resize stages
    no_resize: bool = False  # override resize to None from CLI (tyro-friendly)
    recompute: bool = False  # force recompute of cached dataset statistics
    quit_after_model: bool = False  # stop after model creation for debugging

    # Eval callbacks (each schedules itself via .every; 0 = disabled)
    hist: HistCallback = default(HistCallback())
    chunks: ChunkCallback = default(ChunkCallback())
    viz: FlowPCACallback = default(FlowPCACallback())
    rast: RastCallback = default(RastCallback())
    val_mse: ValMSECallback = default(ValMSECallback())
    synth: SynthVizCallback = default(SynthVizCallback())
    kp3dc: Kp3dcVizCallback = default(Kp3dcVizCallback())

    wandb: cn.Wandb = default(cn.Wandb())


# -- helpers ------------------------------------------------------------------


def infer_model_keys(obs: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Infer image and proprio tokenizer keys from a real observation batch."""
    image_keys = tuple(k.removeprefix("image_") for k in sorted(obs) if k.startswith("image_"))
    proprio_keys = tuple(k.removeprefix("proprio_") for k in sorted(obs) if k.startswith("proprio_"))
    return image_keys, proprio_keys


def _num_tokens(tok_cfg: dict[str, Any]) -> int:
    kwargs = tok_cfg.get("kwargs", {})
    return int(kwargs.get("num_tokens", kwargs.get("num_latents", 0)))


def _leaf_dtypes(tree: Any) -> list[str]:
    return sorted({str(x.dtype) for x in jax.tree.leaves(tree) if hasattr(x, "dtype")})


def _has_tips_subtree(tree: object) -> bool:
    """True if a "tips" submodule (the frozen stacked encoder) lives in the tree."""
    if not isinstance(tree, dict):
        return False
    return "tips" in tree or any(_has_tips_subtree(v) for v in tree.values())


def _build_optimizer_cfg(cfg: Config) -> dict[str, Any]:
    # optax warmup_cosine_decay requires decay_steps > warmup_steps; cap warmup
    # at 10% of the run so short debug runs (--steps 500) neither crash against
    # the default warmup nor spend the whole run warming up.
    warmup_steps = min(cfg.warmup_steps, cfg.steps // 10)
    learning_rate = (
        {
            "name": cfg.lr_schedule,
            "init_value": 0.0,
            "peak_value": cfg.lr,
            "warmup_steps": warmup_steps,
            **({"decay_steps": cfg.steps} if cfg.lr_schedule == "cosine" else {}),
        }
        if warmup_steps > 0
        else cfg.lr
    )
    frozen = list(cfg.frozen_keys)
    # Freeze the pretrained stacked TIPS trunk: freeze_weights routes matched
    # params to optax.set_to_zero(), which drops the whole adamw update (grads
    # AND weight decay), so decay can't shrink the loaded weights. "*tips*"
    # matches the nested "tips" submodule path inside the tokenizer.
    v = cfg.model.vision
    if v.stacked and v.stacked_encoder == "tips" and v.stacked_freeze and "*tips*" not in frozen:
        frozen.append("*tips*")
    return {
        "learning_rate": learning_rate,
        "weight_decay": cfg.weight_decay,
        "clip_gradient": cfg.clip_gradient,
        "frozen_keys": frozen or None,
    }


def _align_batch_size(batch_size: int, device_count: int) -> int:
    if batch_size < device_count:
        return device_count
    return (batch_size // device_count) * device_count


def per_embodiment_metrics(batch: dict[str, Any], update_info: dict[str, Any]) -> dict[str, float]:
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


def shard_batch(batch: Any, mesh: Mesh) -> Any:
    """Shard a host-local batch across the data axis."""
    return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))


def make_data_cfg(mix: str, batch_size: int, loader: Loader, recompute: bool = False) -> cn.Train:
    """Build a Train config for a specific loader."""
    return cn.Train(
        data=cn.Dataset(
            mix=DataSourceE[mix],
            loader=replace(loader, global_batch_size=batch_size),
            recompute=recompute,
        ),
        seed=42,
        verbosity=0,
    )


def _save_path(cfg: Config) -> str:
    if cfg.save_dir is None:
        raise ValueError("save_dir is None")
    return str((Path(cfg.save_dir).expanduser() / cfg.wandb.project / (cfg.wandb.group or "") / cfg.name).resolve())


# -- main ---------------------------------------------------------------------


def main(cfg: Config) -> None:
    initialize_compilation_cache()
    devices = jax.devices()
    mesh = Mesh(devices, axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    print(Rule("XFlowHead + CrossFormerModel: bundled actions", style="bold magenta"))
    print(f"  backend={jax.default_backend()} devices={len(devices)}")
    if cfg.batch_size % len(devices) != 0:
        raise ValueError(f"batch_size={cfg.batch_size} must be divisible by devices={len(devices)}")

    if cfg.model.vision.use_dino:
        from crossformer.model.components.dino_encoder import shard_dino

        shard_dino(replicated_sharding, model_id=cfg.model.vision.dino_model_id)
        print("  dino: state replicated across devices")

    max_h = cfg.horizon
    run = cfg.wandb.initialize(cfg)

    # Load data
    print(Rule("loading data"))
    from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory

    _apply_fd_limit(512**2)
    effective_resize = None if cfg.no_resize else cfg.resize
    train_cfg = make_data_cfg(cfg.mix, cfg.batch_size, cfg.train_loader, recompute=cfg.recompute)
    eval_cfg = make_data_cfg(
        cfg.mix,
        _align_batch_size(min(cfg.batch_size, cfg.eval_batch_size), len(devices)),
        Loader(
            use_grain=True,
            shuffle_buffer=1,
            threads_traj_transform=16,
            threads_traj_read=16,
            threads_frame_transform=16,
            prefetch=8,
        ),
        recompute=cfg.recompute,
    )
    dataset = GrainDataFactory(mp=cfg.mp, rotate=cfg.rotate, resize=effective_resize, batches=cfg.batches).make(
        train_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=True
    )
    if cfg.batches is not None:
        print(f"  [bold yellow]overfit mode: cycling first {cfg.batches} batch(es) forever[/]")
    dsit = iter(dataset.dataset)
    example_batch = next(dsit)
    eval_dataset = GrainDataFactory(
        mp=0,
        shuffle=False,
        mask_slot=False,
        shuffle_slot=False,
        imaug=True,
        rotate=cfg.rotate,
        resize=effective_resize,
    ).make(eval_cfg, shard_fn=partial(shard_batch, mesh=mesh), train=False)
    print(spec(example_batch))
    inferred_image_keys, inferred_proprio_keys = infer_model_keys(example_batch["observation"])
    obs_keys: tuple[str, ...] = ()
    per_device_batch = cfg.batch_size // len(devices)
    image_shapes = {k: tuple(v.shape) for k, v in example_batch["observation"].items() if k.startswith("image_")}
    image_dtypes = {k: str(v.dtype) for k, v in example_batch["observation"].items() if k.startswith("image_")}
    print(f"  per_device_batch: {per_device_batch}")
    print(f"  eval_batch_size: {eval_cfg.data.loader.global_batch_size}")
    print(f"  image_keys: {inferred_image_keys}")
    print(f"  input image shapes: {image_shapes}")
    print(f"  input image dtypes: {image_dtypes}")
    print(f"  proprio_keys (available): {inferred_proprio_keys}")

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
    cfg.model.window = max_w
    cfg.model.image_keys = inferred_image_keys
    print("inferred image_keys: ")
    print(f"{cfg.model.image_keys}")
    cfg.model.proprio_keys = ()
    cfg.model.xflow.max_dofs = max_a
    cfg.model.xflow.max_horizon = max_h
    cfg.model.xflow.use_guidance = cfg.use_guidance
    cfg.model.xflow.guidance_input_dim = None if guide_example is None else guide_example.shape[-1]
    example_obs = flatten_obs(
        example_batch["observation"],
        obs_keys,
        view_mask=example_batch.get("mask", {}).get("view"),
        state=example_batch.get("state"),
        mask=example_batch.get("mask"),
    )

    # With obs_keys empty, example_obs carries every observation leaf verbatim,
    # including the stacked "image" (B, W, V, H, W, C) and the injected view_mask
    # the stacked tokenizer needs. The image_* union below is a no-op on the
    # stacked path (infer_model_keys finds no image_* keys) but keeps the legacy
    # named-key path working.
    init_obs = dict(example_obs)
    init_obs |= {
        k: v
        for k, v in example_batch["observation"].items()
        if any(k == f"image_{name}" for name in cfg.model.image_keys)
    }
    print(f"  image keys in init_obs: {[k for k in init_obs if 'image' in k or 'depth' in k]}")
    init_batch = {
        "observation": init_obs,
        "task": example_batch.get("task", {"pad_mask_dict": {}}),
    }
    model_cfg = cfg.model.create()
    model_cfg["optimizer"] = _build_optimizer_cfg(cfg)
    model_spec = model_cfg["model"]
    obs_tok_cfg = model_spec["observation_tokenizers"]
    task_tok_cfg = model_spec["task_tokenizers"]
    readouts = model_spec["readouts"]
    obs_tokens = sum(_num_tokens(tok) for tok in obs_tok_cfg.values())
    task_tokens = sum(_num_tokens(tok) for tok in task_tok_cfg.values())
    readout_tokens = sum(int(v) for v in readouts.values())
    attn_tokens = obs_tokens + task_tokens + readout_tokens
    head_spec = model_spec["heads"]["action"]
    print(Rule("model diagnostics"))
    print(f"  trunk: {cfg.model.trunk}" + (f" (latents={cfg.model.bela_latents})" if cfg.model.trunk == "bela" else ""))
    print(f"  attention tokens: {attn_tokens} (obs={obs_tokens} task={task_tokens} readout={readout_tokens})")
    print(f"  hidden dim: {model_spec['token_embedding_size']}")
    print(f"  transformer layers: {model_spec['transformer_kwargs']['num_layers']}")
    print(f"  transformer heads: {model_spec['transformer_kwargs']['num_attention_heads']}")
    print(f"  head: {head_spec['name']} ({cfg.model.head_type})")
    if cfg.model.vision.stacked:
        enc_name = cfg.model.vision.stacked_encoder
        variant = f" {cfg.model.vision.tips_variant}" if enc_name == "tips" else ""
        print(
            f"  vision: stacked {enc_name}{variant} (freeze={cfg.model.vision.stacked_freeze})"
            f" — [yellow]vision.encoder/use-film ignored; select via --model.vision.stacked-encoder[/]"
        )
        if enc_name not in ("tips", "dino") and cfg.model.vision.stacked_freeze:
            print(
                f"  [red]warning: {enc_name} has no pretrained weights — freeze=True trains on random"
                f" features; pass --model.vision.no-stacked-freeze[/]"
            )
    else:
        print(f"  vision: per-camera {cfg.model.vision.encoder} (film={cfg.model.vision.use_film})")
    if cfg.model.head_type == "xflow":
        print(f"  xflow self-attend layers: {head_spec['kwargs']['num_self_attend_layers']}")
        print(f"  xflow head blocks: {head_spec['kwargs']['num_blocks']}")
    elif cfg.model.head_type == "pio":
        depth = head_spec["kwargs"]["num_blocks"] * head_spec["kwargs"]["num_self_attend_layers"]
        m = head_spec["kwargs"]["num_act_latents"]
        if m > 0:
            print(f"  pio act_latents: {m}  fuse layers: {head_spec['kwargs']['num_fuse_layers']}")
        else:
            print(f"  pio latents: {head_spec['kwargs']['num_latents']}  process depth: {depth}")
    wandb.config.update(
        {
            "example_batch_spec": spec(example_batch),
            "obs_keys": obs_keys,
            "max_a": max_a,
            **model_cfg,
        },
        allow_val_change=True,
    )

    rng = jax.random.PRNGKey(42)
    init_rng, train_rng, pred_rng = jax.random.split(rng, 3)

    model_cls = BELAModel if cfg.model.trunk == "bela" else CrossFormerModel
    model = model_cls.from_config(
        model_cfg,
        init_batch,
        text_processor=None,
        verbose=cfg.verbose,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
    )
    if cfg.quit_after_model:
        print("quit_after_model=True; stopping after model creation")
        return

    # Load pretrained TIPS weights into every "tips" subtree (stacked encoder).
    # The optimizer freezes these params (see _build_optimizer_cfg), so this is
    # the only place they get their pretrained values.
    if _has_tips_subtree(model.params):
        loaded = load_tips_params(unfreeze(cast(Any, model.params)), variant=cfg.model.vision.tips_variant)
        model = cast(Any, model).replace(params=loaded)
        print(f"  tips: loaded pretrained '{cfg.model.vision.tips_variant}' weights")

    model = cast(Any, model).replace(
        params=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.params),
        example_batch=jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), model.example_batch),
    )
    n_params = sum(x.size for x in jax.tree.leaves(model.params))
    param_dtypes = _leaf_dtypes(model.params)
    print(f"  params: {n_params:,}")
    print(f"  param dtypes: {param_dtypes}")
    print(f"  heads: {list(model.module.heads.keys())}")
    effective_frozen = model.config["optimizer"].get("frozen_keys")
    if effective_frozen:
        print(f"  frozen_keys: {effective_frozen}")
    wandb.config.update({"n_params": n_params}, allow_val_change=True)

    # Guidance config sanity check
    if cfg.use_guidance:
        assert guide_example is not None
        print(Rule("guidance encoder"))
        print(f"  guide_keys={cfg.guide_keys} shape={guide_example.shape}")

    # Optimizer + state
    params = model.params
    tx, lr_callable, param_norm_callable = cast(
        tuple[Any, Any, Any], create_optimizer(params, **model.config["optimizer"])
    )
    print(Rule("optimizer"))
    print(f"  config: {model.config['optimizer']}")
    print(f"  tx: {tx}")
    state = TrainState.create(model=model, tx=tx, rng=train_rng)
    dof_weights = load_loss_weights(cfg.loss_weights) if cfg.loss_weights else None
    if dof_weights is not None:
        print(f"  loss_weights: {cfg.loss_weights} (non-unit dofs: {int((dof_weights != 1.0).sum())})")
    train_step = make_train_step(
        model.module, lr_callable, param_norm_callable, dof_weights=dof_weights, subtree_norms=cfg.subtree_norms
    )

    # Checkpointing
    if cfg.save_dir is not None:
        save_dir = _save_path(cfg)
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        print(f"  save_dir: {save_dir}")
        save_callback = SaveCallback(save_dir)
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        print("  [dim]no save_dir — checkpoints disabled[/]")

    # dataset.dataset_statistics is the JSON-serialized form the denormalizer
    # reads; dataset.statistics is the raw DatasetStatistics (has .unnormalize)
    # that SynthVizCallback needs via ctx.stats.
    eval_loop = EvalLoop(
        loader=eval_dataset.dataset,
        callbacks=[cfg.hist, cfg.chunks, cfg.viz, cfg.rast, cfg.val_mse, cfg.synth, cfg.kp3dc],
        denorm=ActionBatchDenormalizer(dataset.dataset_statistics),
        obs_keys=obs_keys,
        pred_rng=pred_rng,
        stats=dataset.statistics,
        use_guidance=cfg.use_guidance,
        guide_keys=cfg.guide_keys,
        wandb_log=cfg.wandb.log,
    )

    # Train
    print(Rule("training"))

    losses = []
    timer = Timer()
    lowdim_rng = np.random.default_rng(42)
    guide_rng = np.random.default_rng(43)
    for step in tqdm(range(cfg.steps)):
        timer.tick("total")
        with timer("dataset"):
            batch = next(dsit)
            obs = flatten_obs(
                batch["observation"],
                obs_keys,
                view_mask=batch.get("mask", {}).get("view"),
                state=batch.get("state"),
                mask=batch.get("mask"),
            )
            task = batch.get("task", {"pad_mask_dict": {}})
            pad_mask = obs["timestep_pad_mask"]
            lowdim_active = True
            # if cfg.lowdim_drop_prob > 0.0 and lowdim_rng.random() < cfg.lowdim_drop_prob:
            # raise NotImplementedError("lowdim_drop_prob > 0 is not implemented yet")
            # obs = zero_lowdim_obs(obs, obs_keys)
            # lowdim_active = False

            guide_input = None
            if cfg.use_guidance:
                guide_input = lookup_guide(batch, cfg.guide_keys)
                if cfg.guidance_drop_prob > 0.0 and guide_rng.random() < cfg.guidance_drop_prob:
                    guide_input = None

            actions, dof_ids, chunk_steps, view_ids, mask_act = extract_bundled_actions(batch, max_h)

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
                view_ids=view_ids,
                mask_act=mask_act,
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
            cfg.wandb.log(
                {
                    "training": update_info,
                    "timer": timer.get_average_times(),
                    "lowdim_active": lowdim_active,
                    "guidance_active": guide_input is not None,
                    **embody_metrics,
                },
                step=step,
            )
        eval_loop(model, state.model.params, step, is_last=step == cfg.steps - 1)

        if (step + 1) % cfg.save_interval == 0 and save_dir is not None:
            with timer("ckpt"):
                save_callback(state, step + 1)

    # Final checkpoint
    if save_dir is not None:
        save_callback(state, cfg.steps)
        save_callback.wait()

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
