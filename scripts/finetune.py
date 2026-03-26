from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from functools import partial
import logging
from pathlib import Path
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

from crossformer import cn
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.callbacks import DTWVizCallback, InspectCallback, PCAVizCallback, SaveCallback
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
import wandb

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
        if "k3ds" in batch["action"]:
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
            if head_name not in batch["action"]:
                cfg.vprint(f"[DEBUG] Skipping head `{head_name}'")
                continue
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

    log.info("Setting up visualization callbacks")

    pca_viz_cb = PCAVizCallback(
        flow_key=("pred_flow",),
        base_key=("gt_action",),
    )

    dtw_viz_cb = DTWVizCallback(
        flow_key=("pred_flow",),
        base_key=("gt_action",),
        joint_idx_to_plot=0,
        band_radius=15,
    )
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
            log.info(f"Evaluating at step {i}...")
            try:
                viz_batch = next(dsit)
                viz_rng = jax.random.fold_in(rng, i)
                bound = model.module.bind({"params": train_state.model.params}, rngs={"dropout": viz_rng})

                embeddings = bound.crossformer_transformer(
                    viz_batch["observation"],
                    viz_batch["task"],
                    viz_batch["observation"]["timestep_pad_mask"],
                    train=False,
                )

                flow_head_name = next(
                    (
                        name
                        for name, head in bound.heads.items()
                        if hasattr(head, "flow_steps") and name in viz_batch["action"]
                    ),
                    None,
                )

                if flow_head_name is not None:
                    pred_flow = bound.heads[flow_head_name].predict_action(
                        embeddings, train=False, rng=viz_rng, sample_shape=(1,)
                    )
                    pred_flow = jax.device_get(pred_flow)

                    if pred_flow.shape[0] == 1:
                        pred_flow = pred_flow.squeeze(0)

                    gt_action = jax.device_get(viz_batch["action"][flow_head_name])
                    gt_action = gt_action[..., :7]

                    viz_payload = {
                        "pred_flow": pred_flow,
                        "gt_action": gt_action,
                    }

                    pca_frames = pca_viz_cb(viz_payload)
                    pca_out_path = Path(f"/tmp/flow_pca_step_{i}.gif")
                    pca_viz_cb.save(pca_frames, pca_out_path, fps=10)

                    dtw_frames = dtw_viz_cb(viz_payload)
                    dtw_out_path = Path(f"/tmp/flow_dtw_step_{i}.png")
                    dtw_viz_cb.save(dtw_frames, dtw_out_path)

                    wandb.log(
                        {
                            "eval/flow_pca": wandb.Video(str(pca_out_path), fps=10, format="gif"),
                            "eval/flow_dtw": wandb.Image(str(dtw_out_path)),
                        },
                        step=i,
                    )
                    log.info("Successfully logged PCA GIF and DTW Plot to WandB!")

            except Exception as e:
                log.exception(f"VizCallback failed at step {i}: {e}")

        if (i + 1) % cfg.save_interval == 0 and save_dir is not None:
            cfg.vprint("Saving checkpoint...")
            with timer("ckpt"):
                save_callback(train_state, i + 1)


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
