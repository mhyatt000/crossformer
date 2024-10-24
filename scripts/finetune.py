import datetime
from functools import partial
import os
from pprint import pprint

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm

from crossformer.data.oxe import ActionDim, HEAD_TO_DATASET
from crossformer.data.dataset import make_interleaved_dataset, make_single_dataset
from crossformer.data.oxe import make_oxe_dataset_kwargs_and_weights
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.utils.jax_utils import initialize_compilation_cache
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.train_callbacks import SaveCallback, ValidationCallback
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
from crossformer.viz.utils import SequenceViz
import wandb

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

default_config_file = os.path.join(
    os.path.dirname(__file__), "configs/finetune_config.py"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    logging.info(
        f"""
        CrossFormer Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.oxe_kwargs.data_mix}
        Data dir: {FLAGS.config.dataset_kwargs.oxe_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    assert (
        FLAGS.config.batch_size % len(devices) == 0
    ), f"Batch size ({FLAGS.config.batch_size}) must be divisible by the number of devices ({len(devices)})"

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
    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        # name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    #########
    #
    # Load Pretrained model + optionally modify config
    #
    #########

    if not FLAGS.config.debug:
        pretrained_model = CrossFormerModel.load_pretrained(
            FLAGS.config.pretrained_path,
            step=FLAGS.config.pretrained_step,
        )
        flat_config = flax.traverse_util.flatten_dict(
            pretrained_model.config, keep_empty_nodes=True
        )
        for d_key in flax.traverse_util.flatten_dict(
            FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
        ):
            for c_key in list(flat_config.keys()):
                if ".".join(c_key).startswith(".".join(d_key)):
                    print(f"Deleting {'.'.join(c_key)}")
                    del flat_config[c_key]

        config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
        config.update(FLAGS.config.get("update_config", ConfigDict()))
        config = config.to_dict()
        check_config_diff(config, pretrained_model.config)

    #########
    #
    # Setup Data Loader
    #
    #########

    if FLAGS.config.debug:  # only for debugging when model is not needed
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

    original = False
    if original:
        dataset = make_single_dataset(
            FLAGS.config.dataset_kwargs,
            traj_transform_kwargs=FLAGS.config.traj_transform_kwargs,
            frame_transform_kwargs=FLAGS.config.frame_transform_kwargs,
            train=True,
        )
        train_data_iter = (
            dataset.repeat()
            .unbatch()
            .shuffle(FLAGS.config.shuffle_buffer_size)
            .batch(FLAGS.config.batch_size)
            .iterator()
        )
        train_data_iter = map(process_batch, train_data_iter)

    else:
        # load datasets
        if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
            # create dataset_kwargs_list from oxe_kwargs
            (
                FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
                FLAGS.config.dataset_kwargs["sample_weights"],
            ) = make_oxe_dataset_kwargs_and_weights(
                **FLAGS.config.dataset_kwargs["oxe_kwargs"]
            )
            del FLAGS.config.dataset_kwargs["oxe_kwargs"]

        FLAGS.config.dataset_kwargs.batch_size //= jax.process_count()
        for l in FLAGS.config.dataset_kwargs.dataset_kwargs_list:
            l["skip_norm_keys"] = ["proprio_bimanual", "proprio_mano"]

        print(
            FLAGS.config.dataset_kwargs.dataset_kwargs_list[0].get("skip_norm_keys", [])
        )
        # pprint(FLAGS.config.dataset_kwargs)
        dataset = make_interleaved_dataset(**FLAGS.config.dataset_kwargs, train=True)

        train_data_iter = map(
            shard,
            map(
                process_batch,
                dataset.iterator(prefetch=FLAGS.config.prefetch_num_batches),
            ),
        )

    example_batch = next(train_data_iter)

    spec = lambda xtree: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), xtree)
    pprint(spec(example_batch))

    # print(dataset.statistics)

    #########
    #
    # Load Pretrained Model
    #
    #########

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = CrossFormerModel.from_config(
        config,
        example_batch,
        text_processor,
        rng=init_rng,
        dataset_statistics=dataset.dataset_statistics,
        verbose=True,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model

    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        **FLAGS.config.optimizer.to_dict(),
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

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
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
            config_file.write(FLAGS.config.to_json_best_effort())
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

    @partial(
        jax.jit,
        out_shardings=jax.sharding.PositionalSharding(jax.devices()).replicate(),
    )
    def _val_fn(params, batch, rng):
        train = False

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
        action_metrics["deltas"] = deltas

        return action_loss, action_metrics

    def val_fn(params, batch, rng, train=False):
        """
        calls a child val function to do any operations with the model.
        the visualization cannot be jitted (courtesy of wandb)
        viz is not returned since ValidationCallback gets the mean of the metrics
        """
        action_loss, action_metrics = _val_fn(params, batch, rng)

        deltas = action_metrics.pop("deltas")
        batch["predict"] = deltas  # plot with model deltas as actions
        batch = jax.tree.map(lambda x: jnp.asarray(x), batch)

        use_mano = any(
            [
                x["name"] in HEAD_TO_DATASET['mano'] 
                for x in FLAGS.config.dataset_kwargs["dataset_kwargs_list"]
            ]
        )
        if use_mano:
            s = SequenceViz.from_batch(batch, stats=dataset.dataset_statistics).wandb()

        return action_loss, action_metrics

    #########
    #
    # Build validation callback
    #
    #########

    if original:
        if FLAGS.config.modality == "image_conditioned":
            modes_to_evaluate = ["image_conditioned"]
        elif FLAGS.config.modality == "text_conditioned":
            modes_to_evaluate = ["text_conditioned"]
        elif FLAGS.config.modality == "multimodal":
            modes_to_evaluate = ["image_conditioned", "text_conditioned"]
        else:
            modes_to_evaluate = ["base"]

        dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

        val_callback = ValidationCallback(
            loss_fn=val_fn,  #  loss_fn,
            process_batch_fn=process_batch,
            text_processor=text_processor,
            val_dataset_kwargs_list=dataset_kwargs_list,
            dataset_kwargs=FLAGS.config,
            modes_to_evaluate=modes_to_evaluate,
            **FLAGS.config.val_kwargs,
        )

    else:
        val_datasets_kwargs_list, _ = filter_eval_datasets(
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
            FLAGS.config.get("eval_datasets", ()),
        )
        val_callback = ValidationCallback(
            loss_fn=val_fn,  # loss_fn,
            process_batch_fn=lambda batch: shard(process_batch(batch)),
            text_processor=text_processor,
            val_dataset_kwargs_list=val_datasets_kwargs_list,
            dataset_kwargs=FLAGS.config.dataset_kwargs,
            **FLAGS.config.val_kwargs.to_dict(),
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
        actions = actions[: FLAGS.config.rollout_kwargs.num_envs, -1, :4, :]
        return actions

    use_rollout = FLAGS.config.rollout_kwargs.use_rollout
    if use_rollout:
        import simpler_env as simpler
        from simpler_utils import EvalCallback, mk_envs

        tasks = [e for e in simpler.ENVIRONMENTS if "widowx" in e]
        # replicates a few times
        tasks = tasks
        venv = mk_envs(tasks, FLAGS.config.rollout_kwargs.num_envs)
        instructions = venv.env_method("get_language_instruction")

    def transform(batch):
        # zeros = jax.tree.map(lambda arr: jnp.zeros(arr), gapspec)
        batch["observation"]["timestep_pad_mask"] = batch["observation"].pop("pad_mask")

        zeros = jax.tree.map(
            lambda arr: jnp.zeros(
                (
                    FLAGS.config.dataset_kwargs.batch_size
                    - FLAGS.config.rollout_kwargs.num_envs,
                    *arr.shape[1:],
                )
            ),
            batch,
        )
        batch = jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), batch, zeros)

        _instruct = instructions + [
            ""
            for _ in range(
                FLAGS.config.dataset_kwargs.batch_size
                - FLAGS.config.rollout_kwargs.num_envs
            )
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
            batch_size=FLAGS.config.rollout_kwargs.num_envs,
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
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i) % FLAGS.config.eval_interval == 0:  # eval on i=0 for comparison
            logging.info("Evaluating...")

            with timer("val"):
                val_metrics = val_callback(train_state, i + 1)
                SequenceViz.flush(i, limit=32)
                wandb_log(val_metrics, step=i)

            if use_rollout:
                with timer("rollout"):
                    evals = eval_callback(i)
                    wandb_log({"eval": evals}, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
