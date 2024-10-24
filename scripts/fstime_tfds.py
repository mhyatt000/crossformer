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

from crossformer.data.dataset import make_interleaved_dataset, make_single_dataset
from crossformer.data.oxe import (
    ActionDim,
    HEAD_TO_DATASET,
    make_oxe_dataset_kwargs_and_weights,
)
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

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = {}
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )


if __name__ == "__main__":
    app.run(main)
