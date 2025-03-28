import datetime
from typing import *
from functools import partial
import os

from absl import logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax._src.util import tuple_insert
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
from rich.pretty import pprint
import tensorflow as tf
import tqdm
import tyro

from crossformer import cn
from crossformer.data.dataset import make_interleaved_dataset, make_single_dataset
from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET

# make_oxe_dataset_kwargs_and_weights,
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.train_callbacks import (
    SaveCallback,
    ValidationCallback,
    VisCallback,
)
from crossformer.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    filter_eval_datasets,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)
import wandb

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass


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
        Data dir: {cfg.dataset.loc}
        Task Modality: {cfg.modality}
        Finetuning Mode: {cfg.optimizer.mode}

        # Devices: {jax.device_count()}
        Batch size: {cfg.dataset.batch_size} ({cfg.dataset.batch_size // len(devices) } per device)
        # Steps: {cfg.max_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    msg = f"Batch size ({cfg.dataset.batch_size}) must be divisible by the number of devices ({len(devices)})"
    assert cfg.dataset.batch_size % len(devices) == 0, msg

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

    #########
    #
    # Setup WandB
    #
    #########

    # name = format_name_with_config( FLAGS.name, cfg.asdict())
    time = datetime.datetime.now(
        cst := datetime.timezone(datetime.timedelta(hours=-6))
    ).strftime("%m%d")
    wandb_id = f"{cfg.name}_{time}"
    logging.warning(f"TODO use {cfg.wandb} config")
    wandb.init(
        config=cfg.asdict(),
        id=wandb_id,
        # name=cfg.name,
        mode=cn.wab.WandbMode.DISABLED.value if cfg.debug else None,
        # TODO handle internally with cn factory:
        # mode=cfg.wandb.mode(cfg.debug).value
        **cfg.wandb.asdict(),
    )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    if cfg.debug is False and cfg.pretrained_path is not None:
        pretrained_model = CrossFormerModel.load_pretrained(
            cfg.pretrained_path,
            step=cfg.pretrained_step,
        )
        flat_config = flax.traverse_util.flatten_dict(
            pretrained_model.config, keep_empty_nodes=True
        )

        # delete keys from config
        for d_key in cn.BasicDelete().expand():  # list of x.y.z

            for c_key in list(flat_config.keys()):
                # deletes all leaf nodes in cfg.delete
                if ".".join(c_key).startswith(".".join(d_key)):
                    print(f"Deleting {'.'.join(c_key)}")
                    del flat_config[c_key]

        config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
        config.update(cfg.get("update", ConfigDict()))
        config = config.to_dict()
        check_config_diff(config, pretrained_model.config)

    #########
    #
    # Setup Data Loader
    #
    #########

    if cfg.debug:  # only for debugging when model is not needed
        config = {"text_processor": None}

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    # load datasets

    # create dataset_kwargs_list
    data_weights: List[Tuple[str, float]] = cfg.data.value.flatten()
    data_prep: List[cn.DataPrep] = [
        cn.DataPrep(name=d, weight=w) for d, w in data_weights
    ]

    # TODO rename local_batch_size
    cfg.dataset.batch_size //= jax.process_count()

    # for l in cfg.dataset.dataset_kwargs_list:
    # l["skip_norm_keys"] = ["proprio_bimanual", "proprio_mano"]

    dataset = make_interleaved_dataset(**cfg.dataset, train=True)

    train_data_iter = map(
        shard,
        map(
            process_batch,
            dataset.iterator(prefetch=cfg.prefetch_num_batches),
        ),
    )

    example_batch = next(train_data_iter)

    spec = lambda xtree: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), xtree)
    pprint(spec(example_batch))

    # print(dataset.statistics)

    quit()

    #########
    #
    # Load Pretrained Model
    #
    #########

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
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

    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    params = model.params
    if cfg.optimizer.frozen_keys is None:
        cfg.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **cfg.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        model=model,
        tx=tx,
        rng=rng,
    )

    #########
    #
    # Save all metadata
    #
    #########

    if cfg.save_dir is not None:
        save_dir = tf.io.gfile.join(
            cfg.save_dir,
            cfg.wandb.project,
            cfg.wandb.group or "",
            wandb_id,
        )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        save_callback = SaveCallback(save_dir)

        # Add window_size to top of config, to make eval easier
        new_config = ConfigDict(model.config)
        new_config["window_size"] = example_batch["observation"][
            "timestep_pad_mask"
        ].shape[1]
        model = model.replace(config=new_config)

        # Save finetuning config since it's not saved by SaveCallback, i.e. as part of model.save_pretrained()
        with tf.io.gfile.GFile(
            tf.io.gfile.join(save_dir, "finetune_config.json"), "w"
        ) as config_file:
            config_file.write(cfg.to_json_best_effort())
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree.map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

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
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
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

    # @partial(
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
        mano_head = bound_module.heads["mano"]
        deltas = mano_head.predict_action(
            transformer_embeddings,
            train=train,
            rng=rng,
        )
        action_metrics["vis"] = {"deltas": deltas}
        return action_loss, action_metrics

    #########
    #
    # Build validation callback
    #
    #########

    val_datasets_kwargs_list, _ = filter_eval_datasets(
        cfg.dataset["dataset_kwargs_list"],
        cfg.dataset["sample_weights"],
        cfg.get("eval_datasets", ()),
    )
    val_callback = VisCallback(
        loss_fn=val_fn,  # loss_fn,
        process_batch_fn=lambda batch: shard(process_batch(batch)),
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset=cfg.dataset,
        stats=dataset.dataset_statistics,
        **cfg.val_kwargs.to_dict(),
    )

    ########
    #
    # SET UP MODEL ROLLOUTS (from improve & SIMPLER)
    #
    ########

    def _model_step(params, batch, rng, train=False):
        """for evaluation in env"""

        # modified for crossformer from octo
        print(spec(batch))

        """
        actions = model.sample_actions(
            batch,
            task,
            model.dataset_statistics['bridge_dataset'],
            head_name='single_arm',
            rng=rng,
        )[0, :, : 7]
        """

        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.crossformer_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        actions = bound_module.heads["single_arm"](
            transformer_embeddings,  # doesnt need rng since its not diffusion
            train=train,
        )

        return actions

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(dp_sharding),
        out_shardings=(replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        # donate_argnums=0,
    )
    def model_step(batch):
        actions = _model_step(train_state.model.params, batch, train_state.rng)
        # act_horizon is 4 for single_arm
        # we act on the last obs horizon
        actions = actions[: cfg.rollout_kwargs.num_envs, -1, :4, :]
        return actions

    use_rollout = cfg.rollout_kwargs.use_rollout
    if use_rollout:
        import simpler_env as simpler
        from simpler_utils import EvalCallback, mk_envs

        tasks = [e for e in simpler.ENVIRONMENTS if "widowx" in e]
        # replicates a few times
        tasks = tasks
        venv = mk_envs(tasks, cfg.rollout_kwargs.num_envs)
        instructions = venv.env_method("get_language_instruction")

    def transform(batch):
        # zeros = jax.tree.map(lambda arr: jnp.zeros(arr), gapspec)
        batch["observation"]["timestep_pad_mask"] = batch["observation"].pop("pad_mask")

        zeros = jax.tree.map(
            lambda arr: jnp.zeros(
                (
                    cfg.dataset.batch_size - cfg.rollout_kwargs.num_envs,
                    *arr.shape[1:],
                )
            ),
            batch,
        )
        batch = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), batch, zeros)

        _instruct = instructions + [
            "" for _ in range(cfg.dataset.batch_size - cfg.rollout_kwargs.num_envs)
        ]
        batch["task"] = {"language_instruction": [i.encode("utf-8") for i in _instruct]}
        batch["dataset_name"] = "bridge_dataset"  # dummy variable

        batch = shard(process_batch(batch))
        return batch

    if use_rollout:
        from improve.fm.oxes import OXESimplerInference, PolicyStepper

        stepper = PolicyStepper(
            model_type="func",
            dataset_id="bridge_dataset",  # or google dataset
            func=model_step,
            transform=transform,
        )

        oxes = OXESimplerInference(
            stepper,
            batch_size=cfg.rollout_kwargs.num_envs,
            image_size=224,
        )
        oxes.reset(instructions)

        def og_step(obs):
            raw, act = oxes.step(obs)
            return act

        eval_callback = EvalCallback(venv, og_step)

    #########
    #
    # Train loop
    #
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(range(cfg.max_steps), total=cfg.max_steps, dynamic_ncols=True):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % cfg.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i) % cfg.eval_interval == 0:  # eval on i=0 for comparison
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            if use_rollout:
                with timer("rollout"):
                    evals = eval_callback(i)
                    wandb_log({"eval": evals}, step=i)

        if (i + 1) % cfg.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
