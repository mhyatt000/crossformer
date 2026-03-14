from __future__ import annotations

from functools import partial
import logging
from typing import Any

from box import Box
from einops import rearrange
import flax
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import ConfigDict
import optax

# from crossformer.utils import nowarn
import tensorflow as tf
import tqdm
import tyro
import wandb

from crossformer import cn
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.callbacks import SaveCallback
from crossformer.utils.callbacks.flow_viz import (
    _DEFAULT_K,
    FlowVisCallback,
    load_camera_extrinsics,
)
from crossformer.utils.callbacks.inspect import InspectCallback
from crossformer.utils.deco import deprecate
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec, spec
from crossformer.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

log = logging.getLogger(__name__)


def main(cfg: cn.Train) -> None:  # experiment or sweep
    #
    # Setup
    #

    tf.random.set_seed(cfg.seed + jax.process_index())
    tf.config.set_visible_devices([], "GPU")

    cfg.vprint(cfg, level=2)
    initialize_compilation_cache()
    devices = jax.devices()

    log.info(
        f"""
        CrossFormer Finetuning Script
        ======================
        Pretrained model: {cfg.pretrained_path}
        Finetuning Dataset: {cfg.data}
        Data dir: {cfg.data.reader.loc}
        Task Modality: {cfg.modality}
        Finetuning Mode: {cfg.optimizer.mode}

        # Devices: {jax.device_count()}
        Batch size: {cfg.data.gbs} ({cfg.data.bs} per device)
        # Steps: {cfg.steps}
    """
    )

    #
    # Setup Jax Data Parallelism
    #

    msg = f"Batch size ({cfg.data.gbs}) must be divisible by the number of devices ({len(devices)})"
    assert cfg.data.gbs % len(devices) == 0, msg

    mesh = Mesh(jax.devices(), axis_names="batch")
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    @deprecate("let grain do it", strict=False)
    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    run = cfg.wandb.initialize(cfg)

    cfg.vprint("== setup complete ==")
    # quit()  # DEVSTOP: setup

    #
    # Load Pretrained model + optionally modify config
    #

    if cfg.model.debug is False and cfg.pretrained_path is not None:
        pretrained_model = CrossFormerModel.load_pretrained(
            cfg.pretrained_path,
            step=cfg.pretrained_step,
        )
        flat_config = flax.traverse_util.flatten_dict(pretrained_model.config, keep_empty_nodes=True)

        flat_config = cfg.model.delete(flat_config, verbose=cfg.verbosity >= 2)

        config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
        cfg.vprint(cfg.model.create(), level=2)
        config.update(cfg.model.create())
        config = config.to_dict()
        check_config_diff(config, pretrained_model.config)

    cfg.vprint("== pretrained model loaded ==")
    # quit()  # DEVSTOP: pretrained model

    #
    # Setup Data Loader
    #

    if cfg.model.debug:  # only for debugging when model is not needed
        config = {"text_processor": None}
    text_processor = None if config["text_processor"] is None else ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    #
    # Load datasets
    #

    if cfg.data.loader.use_grain:
        from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory

        dataset = GrainDataFactory().make(cfg, shard_fn=shard, train=True)
        _apply_fd_limit(512**2)
        dsit = iter(dataset.dataset)
    else:
        dataset = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS, train=True)
        dsit = dataset.iterator(prefetch=cfg.data.loader.prefetch)
        dsit = map(shard, map(process_batch, dsit))

    log.warning("TODO shard with mp")
    example_batch = next(dsit)
    cfg.vprint(spec(example_batch), level=2)

    callbacks = Box(inspect=InspectCallback(log_every=100))
    _ = callbacks.inspect(example_batch)

    cfg.vprint("== data loaded ==")
    # quit()  # DEVSTOP: data loading

    #
    # Init Model from Config
    #

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    cfg.vprint(config, level=2)
    model = CrossFormerModel.from_config(
        config,
        example_batch,
        text_processor=text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
        verbose=cfg.verbosity >= 2,
    )

    if cfg.pretrained_path is not None:
        merged_params = merge_params(model.params, pretrained_model.params)
        model = model.replace(params=merged_params)
        del pretrained_model

    cfg.vprint("== model initialized ==")
    # quit()  # DEVSTOP: model init

    #
    # Setup Optimizer and Train State
    #

    params = model.params
    if cfg.optimizer.frozen_keys is None:
        cfg.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(params, **cfg.optimizer.create())
    train_state = TrainState.create(model=model, tx=tx, rng=rng)

    #
    # Save all metadata
    #

    if cfg.save_dir is not None:
        save_dir = tf.io.gfile.join(
            str(cfg.save_dir),
            cfg.wandb.project,
            cfg.wandb.group or "",
            cfg.name,
        )
        wandb.config.update({"save_dir": save_dir}, allow_val_change=True)
        log.info(f"Saving to {save_dir}")
        save_callback = SaveCallback(save_dir)

        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"]["timestep_pad_mask"].shape[1]
        model = model.replace(config=new_config)

        log.warning("not saving config to disk")
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        log.warning("save_dir not passed in, not saving checkpoints")

    wandb.config.update({"example_batch_spec": spec(example_batch)}, allow_val_change=True)

    cfg.vprint("== optimizer + state ready ==")
    # quit()  # DEVSTOP: optimizer

    #
    # Define loss, train_step, and eval_step
    #

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.crossformer_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        cfg.vprint("-> patch")
        # quick patch
        unsqueeze = lambda x: jnp.expand_dims(x, axis=1)
        batch["action"] = jax.tree.map(unsqueeze, batch["action"])
        # fix k3ds

        batch["action"]["k3ds"] = rearrange(batch["action"]["k3ds"], "b w h x y -> b w h (x y)")
        squeeze = lambda x: jnp.squeeze(x, axis=-1)
        batch["embodiment"] = jax.tree.map(squeeze, batch["embodiment"])
        # end patch
        cfg.vprint("-> end patch")

        action_loss, action_metrics = 0, {}
        for head_name, head in bound_module.heads.items():
            cfg.vprint(f"[DEBUG] Processing head: {head_name}")
            # if head_name == "single_arm":
            # head_loss, head_metrics = head.loss(embeddings=transformer_embeddings, batch=batch, train=True)
            # else:
            head_loss, head_metrics = head.loss(
                transformer_embeddings,
                batch["action"][head_name],
                batch["observation"]["timestep_pad_mask"],
                action_pad_mask=jnp.ones_like(batch["action"][head_name], dtype=jnp.bool_),
                action_head_mask=batch["embodiment"][head_name],
                train=train,
            )

            # head_sample_fraction = (batch["action_head_masks"][head_name].sum()) / len(batch["action"])
            head_sample_fraction = batch["embodiment"][head_name].mean()
            action_loss += head_loss * head_sample_fraction * head.loss_weight
            action_metrics[head_name] = head_metrics
        action_metrics["total_loss"] = action_loss

        return action_loss, action_metrics

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.model.params, batch, dropout_rng, train=True)
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    def val_fn(params, batch, rng, train=False):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.crossformer_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )

        action_loss, action_metrics = 0, {}
        for head_name, head in bound_module.heads.items():
            head_loss, head_metrics = head.loss(
                transformer_embeddings,
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                action_head_mask=batch["action_head_masks"][head_name],
                train=train,
            )

            head_sample_fraction = (batch["action_head_masks"][head_name].sum()) / len(batch["action"])
            action_loss += head_loss * head_sample_fraction * head.loss_weight
            action_metrics[head_name] = head_metrics

        if "mano" in cfg.model.heads:
            mano_head = bound_module.heads["mano"]
            deltas = mano_head.predict_action(
                transformer_embeddings,
                train=train,
                rng=rng,
            )
            action_metrics["vis"] = {"deltas": deltas}
        return action_loss, action_metrics

    #
    # Build validation callback
    #

    def _flow_viz_iter():
        while True:
            yield next(dsit)

    def flow_eval_step(state: TrainState, batch):
        rng_eval = jax.random.fold_in(state.rng, int(state.step))
        _, metrics = val_fn(state.model.params, batch, rng_eval, train=False)
        vis = metrics.get("vis", {})
        if not isinstance(vis, dict):
            vis = {}
        if "q_flow_steps" not in vis and "deltas" in vis:
            vis["q_flow_steps"] = vis["deltas"]
        return {"text_conditioned": {"vis": vis}}

    flow_viz_callback = object.__new__(FlowVisCallback)
    flow_viz_callback.fps = 8
    flow_viz_callback.max_videos = 2
    flow_viz_callback.num_val_batches = 1
    flow_viz_callback.camera_view = "low"
    flow_viz_callback.camera_intrinsics = _DEFAULT_K.copy()
    flow_viz_callback.camera_R, flow_viz_callback.camera_t = load_camera_extrinsics("low")
    flow_viz_callback.ros_to_opencv = True
    flow_viz_callback.enable_denoise_plots = False
    flow_viz_callback.enable_xyz_image_flow = False
    flow_viz_callback.enable_joint_xyz_pca_flow = False
    flow_viz_callback.enable_part_a = True
    flow_viz_callback.enable_part_b = True
    flow_viz_callback.enable_part_c = True
    flow_viz_callback.robot_pad_gripper = True
    flow_viz_callback.flow_overlay_ft_index = 0
    flow_viz_callback.flow_q_keys = ("q_flow_steps", "joint_flow_steps", "deltas")
    flow_viz_callback.flow_xyz_keys = ("fk_xyz_flow_steps", "xyz_flow_steps", "joints_flow_steps")
    flow_viz_callback.compute_fk_from_q_steps = True
    flow_viz_callback.q_pca_dims = 7
    flow_viz_callback._fk_fn = None
    flow_viz_callback._robot = None
    flow_viz_callback.val_iterators = {"train_stream": _flow_viz_iter()}
    flow_viz_callback.eval_step = flow_eval_step

    log.info("flow_viz callback enabled")

    #
    # Train loop
    #
    cfg.vprint("== starting training ==")
    # quit() # DEVSTOP: before training loop

    timer = Timer()
    for i in tqdm.tqdm(range(cfg.steps), total=cfg.steps, dynamic_ncols=True):
        timer.tick("total")

        with timer("dataset"):
            batch = next(dsit)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        maybe_inspect: dict[str, Any] = {}

        timer.tock("total")

        if (i + 1) % cfg.log_interval == 0:
            update_info = jax.device_get(update_info)
            cfg.wandb.log(
                {
                    "training": update_info,
                    "inspect": maybe_inspect,
                    "timer": timer.get_average_times(),
                },
                step=i,
            )

        if (i) % cfg.eval_interval == 0:
            log.info("Evaluating...")
            with timer("val"):
                flow_vis_metrics = flow_viz_callback(train_state, i + 1)
                if flow_vis_metrics:
                    cfg.wandb.log(flow_vis_metrics, step=i)

        if (i + 1) % cfg.save_interval == 0 and save_dir is not None:
            cfg.vprint("Saving checkpoint...")
            with timer("ckpt"):
                save_callback(train_state, i + 1)


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
