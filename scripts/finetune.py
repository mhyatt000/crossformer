import datetime
from functools import partial
from typing import Any

from absl import logging
from box import Box
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec
from ml_collections import ConfigDict
import optax
from rich.pretty import pprint
import tensorflow as tf
import tqdm
import tyro

from crossformer import cn
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.train_callbacks import SaveCallback
from crossformer.utils.train_callbacks import VisCallback
from crossformer.utils.train_utils import check_config_diff
from crossformer.utils.train_utils import create_optimizer
from crossformer.utils.train_utils import merge_params
from crossformer.utils.train_utils import process_text
from crossformer.utils.train_utils import Timer
from crossformer.utils.train_utils import TrainState
import wandb

# make_oxe_dataset_kwargs_and_weights,

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass


def set_wandb(cfg: cn.Train):
    # name = format_name_with_config( FLAGS.name, cfg.asdict())
    time = datetime.datetime.now(
        cst := datetime.timezone(datetime.timedelta(hours=-6))
    ).strftime("%m%d")
    # wandb_id = f"{cfg.name}_{time}"
    logging.warning(f"TODO use {cfg.wandb} config")
    run = wandb.init(
        config=cfg.asdict(),
        # # id=wandb_id,
        # # name=cfg.name,
        mode=cfg.wandb.mode(cfg.debug),
        **cfg.wandb.asdict(),
    )
    run.name = f"{time}_{run.name}"
    cfg.name = run.name


def wandb_log(info, step):
    wandb.log(flatten_dict(info, sep="/"), step=step)


def main(cfg: cn.Train) -> None:  # experiment or sweep
    pprint(cfg)
    initialize_compilation_cache()
    devices = jax.devices()

    logging.info(
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

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    # make sure each process loads different data
    tf.random.set_seed(cfg.seed + jax.process_index())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #
    # Setup WandB
    #

    set_wandb(cfg)

    #
    # Load Pretrained model + optionally modify config
    #

    if cfg.model.debug is False and cfg.pretrained_path is not None:
        pretrained_model = CrossFormerModel.load_pretrained(
            cfg.pretrained_path,
            step=cfg.pretrained_step,
        )
        flat_config = flax.traverse_util.flatten_dict(
            pretrained_model.config, keep_empty_nodes=True
        )

        flat_config = cfg.model.delete(flat_config)

        config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
        pprint(cfg.model.create())
        config.update(cfg.model.create())
        config = config.to_dict()
        check_config_diff(config, pretrained_model.config)

    #
    # Setup Data Loader
    #

    if cfg.model.debug:  # only for debugging when model is not needed
        config = {"text_processor": None}
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    #
    # load datasets
    #

    dataset = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS, train=True)
    data_iter = dataset.iterator(prefetch=cfg.data.loader.prefetch)
    data_iter = map(shard, map(process_batch, data_iter))

    example_batch = next(data_iter)
    spec = lambda _x: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), _x)
    pprint(spec(example_batch))

    from crossformer.utils.callbacks.inspect import InspectCallback

    callbacks = Box(inspect=InspectCallback(log_every=100))
    _ = callbacks.inspect(example_batch)

    #
    # Load Pretrained Model
    #

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    pprint(config)
    model = CrossFormerModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
        verbose=True,
    )

    if cfg.pretrained_path is not None:
        merged_params = merge_params(model.params, pretrained_model.params)
        model = model.replace(params=merged_params)
        del pretrained_model

    #
    # Setup Optimizer and Train State
    #

    params = model.params
    if cfg.optimizer.frozen_keys is None:
        cfg.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params, **cfg.optimizer.create()
    )
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
        logging.info(f"Saving to {save_dir}")
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        logging.warning("WARNING: not saving config to disk")
        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        # with tf.io.gfile.GFile(
        # tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        # ) as config_file:
        # config_file.write(cfg.serialize())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    wandb.config.update(
        {"example_batch_spec": spec(example_batch)}, allow_val_change=True
    )

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

        action_loss, action_metrics = 0, {}
        for head_name, head in bound_module.heads.items():
            head_loss, head_metrics = head.loss(
                transformer_embeddings,  # action head knows to pull out the "action" readout_key
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                action_head_mask=batch["action_head_masks"][head_name],
                train=train,
            )

            # weight loss by number of samples from each head
            head_sample_fraction = (batch["action_head_masks"][head_name].sum()) / len(
                batch["action"]
            )
            action_loss += head_loss * head_sample_fraction * head.loss_weight
            action_metrics[head_name] = head_metrics
        action_metrics["total_loss"] = action_loss

        return action_loss, action_metrics

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
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

    # @partial(  # cant pass kwargs to jitted function
    # jax.jit,
    # in_shardings=[replicated_sharding, dp_sharding],
    # )
    def val_fn(params, batch, rng, train=False):
        #
        # TODO fix to become part of loss_fn
        # if not train then get mano deltas else none
        #

        # loss calculations
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
                transformer_embeddings,  # action head knows to pull out the "action" readout_key
                batch["action"],
                batch["observation"]["timestep_pad_mask"],
                batch["action_pad_mask"],
                action_head_mask=batch["action_head_masks"][head_name],
                train=train,
            )

            # weight loss by number of samples from each head
            head_sample_fraction = (batch["action_head_masks"][head_name].sum()) / len(
                batch["action"]
            )
            action_loss += head_loss * head_sample_fraction * head.loss_weight
            action_metrics[head_name] = head_metrics

        # eval metrics
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

    # identical dataset kwargs for eval
    dks, _ = cfg.data.kwargs_list(oxe_fns=OXE_STANDARDIZATION_TRANSFORMS)
    val_callback = VisCallback(
        loss_fn=val_fn,  # loss_fn,
        process_batch_fn=lambda batch: shard(process_batch(batch)),
        text_processor=text_processor,
        val_dataset_kwargs_list=dks,
        dataset_kwargs=cfg.data.kwargs(),
        stats=dataset.dataset_statistics,
        val_shuffle_buffer_size=1_000,
        num_val_batches=8,
    )

    #
    # Train loop
    #

    timer = Timer()
    for i in tqdm.tqdm(range(cfg.steps), total=cfg.steps, dynamic_ncols=True):
        timer.tick("total")

        with timer("dataset"):
            batch = next(data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        maybe: dict[str, Any] = callbacks.inspect.every(batch=batch, step=i)
        update_info.update(maybe)

        timer.tock("total")

        if (i + 1) % cfg.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )
            # pprint(update_info)

        if (i) % cfg.eval_interval == 0:  # eval on i=0 for comparison
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            # if cfg.rollout.use:
            # with timer("rollout"):
            # evals = eval_callback(i)
            # wandb_log({"eval": evals}, step=i)

        if (i + 1) % cfg.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
