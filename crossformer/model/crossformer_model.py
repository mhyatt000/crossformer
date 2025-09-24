from functools import partial
import json
import logging
from pathlib import Path
from typing import Any

import flax
from flax import struct
from flax.training import orbax_utils
import jax
from jax import ShapeDtypeStruct
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from orbax import checkpoint as ocp
import tensorflow as tf

from crossformer.data.utils.data_utils import NormalizationType
from crossformer.data.utils.text_processing import TextProcessor
from crossformer.model.components.action_heads import ActionHead
from crossformer.model.crossformer_module import CrossFormerModule
from crossformer.utils.spec import ModuleSpec
from crossformer.utils.typing import Config
from crossformer.utils.typing import Data
from crossformer.utils.typing import Params
from crossformer.utils.typing import PRNGKey
from crossformer.utils.typing import Sequence


def _lookup_path(tree, path):
    node = tree
    for entry in path:
        if isinstance(entry, jax.tree_util.DictKey):
            node = node[entry.key]
        elif isinstance(entry, jax.tree_util.GetAttrKey):
            node = getattr(node, entry.attr)
        elif isinstance(entry, jax.tree_util.SequenceKey):
            node = node[entry.idx]
        else:
            raise TypeError(f"Unsupported path entry: {entry}.")
    return node


def _zeros_from_spec(spec):
    try:
        return np.zeros(spec.shape, spec.dtype)
    except (TypeError, ValueError, AttributeError):
        return jnp.zeros(spec.shape, spec.dtype)


def _apply_sharding(target, sharding):
    if isinstance(sharding, jax.sharding.Sharding):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), target)

    (path_leaves, tree_def) = jax.tree_util.tree_flatten_with_path(target)
    sharding_flat = []
    for path, _ in path_leaves:
        try:
            shard = _lookup_path(sharding, path)
        except (AttributeError, KeyError, IndexError, TypeError) as err:
            raise ValueError("Sharding tree does not match params tree.") from err
        sharding_flat.append(shard)

    placed = []
    for (_, value), shard in zip(path_leaves, sharding_flat):
        placed.append(jax.device_put(value, shard) if shard is not None else value)
    return jax.tree_util.tree_unflatten(tree_def, placed)


# Build your target params abstract as you do now (e.g., via model.init(...)).
# Suppose that's `abstract` (same tree/keys as your live model variables).


def _to_unsharded(x):
    # Drop sharding metadata; keep only shape/dtype
    return ShapeDtypeStruct(x.shape, x.dtype)


def spec(tree):
    return jax.tree.map(ocp.utils.to_shape_dtype_struct, tree)


def restore_params(
    checkpoint_dir: str,
    params_shape,
    step: int | None = None,
    sharding=None,
):
    """Restores a params pytree, optionally placing leaves with ``sharding``."""
    manager = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer())
    # cp = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) # new API
    step = manager.latest_step() if step is None else step
    if step is None:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}.")

    target = jax.tree_util.tree_map(_zeros_from_spec, params_shape)
    # target = tree_map(_to_unsharded, target)

    if sharding is not None:
        target = _apply_sharding(target, sharding)

    _params = manager.restore(step)
    _params["model"]
    print(_params.keys())
    print(_params["model"]["params"].keys())
    return _params["model"]["params"]

    return manager.restore(
        step,
        args=ocp.args.StandardRestore(item=target),
    )

    # use new orbax API for flexible restore
    # beware rough edges
    target = jax.tree_util.tree_map(jnp.zeros_like, params_shape)
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(jax.devices(), ("model",)),
        jax.sharding.PartitionSpec(
            None,
        ),
    )
    doshard = lambda x: jax.device_put(x, sharding)
    target = jax.tree.map(doshard, target)
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, target)

    manager = ocp.CheckpointManager(ckpt_path, ocp.StandardCheckpointer())
    # structure = manager.item_metadata(step)
    # target = jax.tree_util.tree_map(lambda x: np.zeros(x.shape), structure)
    # params = manager.restore(step, args=ocp.args.StandardRestore(abstract))
    params = manager.restore(cpath)
    return


@struct.dataclass
class CrossFormerModel:
    """Recommended way of interacting with CrossFormer models.

    Usage for inference:

        >>> model = CrossFormerModel.load_pretrained(checkpoint_dir)
        >>> tasks = model.create_tasks(texts=["go to the red room"])
        >>> # or tasks = model.create_tasks(goals={"image_primary": goal_images})
        >>> actions = model.sample_actions(observations, tasks, rng=jax.random.PRNGKey(0))
        >>> # Note: these are normalized actions (processed to mean 0 and std 1). To get correct actions
        >>> # for a particular embodiment, you must additionally specify unnormalization statistics.
        >>> # For example, to get actions for one of CrossFormer's pretraining datasets:
        >>> actions = model.sample_actions(observations, tasks, rng=jax.random.PRNGKey(0),
        >>>     unnormalization_statistics=model.dataset_statistics["DATASET_NAME_HERE"]["action"]
        >>> )

    Usage for finetuning:

        >>> model = CrossFormerModel.load_pretrained(checkpoint_dir)
        >>> train_state = crossformer.utils.train_utils.TrainState.create(
            rng=jax.random.PRNGKey(0),
            model=model,
            tx=optax.adamw(...)
        )
        >>> # access params through train_state.model.params
        >>> train_state, metrics = your_update_function(train_state, batch)
        >>> # when it's time to save (note that this only saves the model parameters,
        >>> # not the full optimizer state)
        >>> train_state.model.save_pretrained(step, save_dir)

    Usage for pretraining:

        >>> model = CrossFormerModel.from_config(
                config,
                example_batch,
                text_processor
            )  # initializes params
        >>> # Continue as in finetuning example

    See full usage examples in train.py and finetune.py.

    """

    module: CrossFormerModule = struct.field(pytree_node=False)
    text_processor: TextProcessor = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    example_batch: Data
    dataset_statistics: Data | None

    def create_tasks(
        self, goals: Data | None = None, texts: Sequence[str] | None = None
    ):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update(
                {k: np.ones(v.shape[:1], dtype=bool) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            tasks.update(
                {
                    k: np.zeros((batch_size, *v.shape[1:]), dtype=v.dtype)
                    for k, v in self.example_batch["task"].items()
                    if k not in ("pad_mask_dict", "language_instruction")
                }
            )
            tasks["pad_mask_dict"].update(
                {
                    k: np.zeros(batch_size, dtype=bool)
                    for k in tasks
                    if k != "pad_mask_dict"
                }
            )

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = texts
            tasks["pad_mask_dict"]["language_instruction"] = np.ones(
                len(texts), dtype=bool
            )
        else:
            batch_size = jax.tree_leaves(goals)[0].shape[0]
            tasks["language_instruction"] = [""] * batch_size
            tasks["pad_mask_dict"]["language_instruction"] = np.zeros(
                batch_size, dtype=bool
            )

        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(
                tasks["language_instruction"]
            )
        else:
            del tasks["language_instruction"]

        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks

    @partial(jax.jit, static_argnames=("train",))
    def run_transformer(
        self,
        observations: Data,
        tasks: Data,
        timestep_pad_mask: ArrayLike,
        train: bool = False,
    ):
        """Runs the transformer, but does shape checking on the inputs.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["task"]
            timestep_pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
        """
        _verify_shapes(
            observations,
            "observations",
            self.example_batch["observation"],
            starting_dim=2,
        )
        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

        return self.module.apply(
            {"params": self.params},
            observations,
            tasks,
            timestep_pad_mask,
            train=train,
            method="crossformer_transformer",
        )

    @partial(
        jax.jit,
        static_argnames=("train", "sample_shape", "argmax", "head_name"),
    )
    def sample_actions(
        self,
        observations: Data,
        tasks: Data,
        unnormalization_statistics: Data | None = None,
        normalization_type: NormalizationType = NormalizationType.NORMAL,
        timestep_pad_mask: ArrayLike | None = None,
        train: bool = False,
        argmax: bool = False,
        sample_shape: tuple[int, ...] = (),
        rng: PRNGKey | None = None,
        temperature: float = 1.0,
        head_name: str = "action",
    ):
        """Samples actions from the model. See `action_heads.py` for more info.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            unnormalization_statistics: dict of statistics for unnormalizing actions (must contain "mean",
                "std", and optionally "mask")
            normalization_type: type of normalization applied to the actions
            timestep_pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            ...see `action_heads.py` for the rest of the kwargs.
        Returns:
            actions: (*sample_shape, batch_size, action_horizon, action_dim)
        """
        if timestep_pad_mask is None:
            timestep_pad_mask = observations["timestep_pad_mask"]

        transformer_outputs = self.run_transformer(
            observations, tasks, timestep_pad_mask, train=train
        )
        action_head: ActionHead = self.module.bind({"params": self.params}).heads[
            head_name
        ]
        action = action_head.predict_action(
            transformer_outputs,
            train=train,
            argmax=argmax,
            sample_shape=sample_shape,
            rng=rng,
            temperature=temperature,
            embodiment_action_dim=(
                len(unnormalization_statistics["mean"])
                if unnormalization_statistics is not None
                else None
            ),
        )
        if unnormalization_statistics is not None:
            if normalization_type == NormalizationType.NORMAL:
                mask = unnormalization_statistics.get(
                    "mask",
                    jnp.ones_like(unnormalization_statistics["mean"], dtype=bool),
                )
                action = action[..., : len(mask)]
                action = jnp.where(
                    mask,
                    (action * unnormalization_statistics["std"])
                    + unnormalization_statistics["mean"],
                    action,
                )
            elif normalization_type == NormalizationType.BOUNDS:
                mask = unnormalization_statistics.get(
                    "mask", jnp.ones_like(unnormalization_statistics["p01"], dtype=bool)
                )
                action = action[..., : len(mask)]
                action = jnp.where(
                    mask,
                    (action + 1)
                    * (
                        unnormalization_statistics["p99"]
                        - unnormalization_statistics["p01"]
                    )
                    / 2
                    + unnormalization_statistics["p01"],
                    action,
                )
            else:
                raise ValueError(f"Unknown normalization type: {normalization_type}")
        return action

    @classmethod
    def load_pretrained(
        cls,
        ckpt_path: str,
        step: int | None = None,
    ) -> "CrossFormerModel":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            ckpt_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        if ckpt_path.startswith("hf://"):
            if step:
                raise ValueError(
                    "You can't set config['pretrained_step'] when loading from HuggingFace."
                )
            ckpt_path = _download_from_huggingface(ckpt_path.removeprefix("hf://"))

        # load config
        with Path(ckpt_path).joinpath("config.json").open("r") as f:
            config = json.load(f)
        # load example batch
        with Path(ckpt_path).joinpath("example_batch.msgpack").open("rb") as f:
            example_batch = flax.serialization.msgpack_restore(f.read())

        rep_shape = lambda x: flax.core.pretty_repr(jax.tree_map(lambda y: y.shape, x))
        logging.debug(
            "Model was trained with observations: %s",
            rep_shape(example_batch["observation"]),
        )
        logging.debug(
            "Model was trained with tasks: %s", rep_shape(example_batch["task"])
        )

        # load dataset statistics
        with Path(ckpt_path).joinpath("dataset_statistics.json").open("r") as f:
            dataset_statistics = json.load(f)
            dataset_statistics = jax.tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )

        module = CrossFormerModule.create(**config["model"])

        # infer params shape without actually doing any computation
        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["timestep_pad_mask"],
        )
        params_shape = jax.eval_shape(
            partial(module.init, train=False), jax.random.PRNGKey(0), *init_args
        )["params"]
        params = restore_params(ckpt_path, params_shape, step)

        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None

        return cls(
            module=module,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )

    def save_pretrained(
        self,
        step: int,
        checkpoint_path: str | None = None,
        checkpoint_manager: ocp.CheckpointManager | None = None,
    ):
        """Saves a model, as well as corresponding metadata needed for `load_pretrained`. Takes either a
        pre-existing checkpoint manager (which already knows where to save the checkpoint) or a path to a
        directory to save the checkpoint to.

        Args:
            step (int): Step number.
            checkpoint_path (str, optional): Path to save the checkpoint.
            checkpoint_manager (optional): Checkpoint manager to save the checkpoint.
            params (optional): Params to save. If None, uses self.params.
        """
        if (checkpoint_path is None) == (checkpoint_manager is None):
            raise ValueError(
                "Must provide exactly one of checkpoint_path or checkpoint_manager."
            )
        if checkpoint_manager is None:
            checkpoint_manager = ocp.CheckpointManager(
                checkpoint_path, ocp.PyTreeCheckpointer()
            )
        if checkpoint_path is None:
            checkpoint_path = str(checkpoint_manager._directory)

        # save params
        checkpoint_manager.save(
            step,
            self.params,
            {"save_args": orbax_utils.save_args_from_target(self.params)},
        )

        if jax.process_index() == 0:
            # save config
            config_path = tf.io.gfile.join(checkpoint_path, "config.json")
            if not tf.io.gfile.exists(config_path):
                with tf.io.gfile.GFile(config_path, "w") as f:
                    json.dump(self.config, f)

            # save example batch
            example_batch_path = tf.io.gfile.join(
                checkpoint_path, "example_batch.msgpack"
            )
            if not tf.io.gfile.exists(example_batch_path):
                with tf.io.gfile.GFile(example_batch_path, "wb") as f:
                    f.write(flax.serialization.msgpack_serialize(self.example_batch))

            # save dataset statistics
            dataset_statistics_path = tf.io.gfile.join(
                checkpoint_path, "dataset_statistics.json"
            )
            if not tf.io.gfile.exists(dataset_statistics_path):
                with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
                    json.dump(
                        jax.tree_map(lambda x: x.tolist(), self.dataset_statistics),
                        f,
                    )

    @classmethod
    def from_config(
        cls,
        config: Config,
        example_batch: Data,
        text_processor: Any | None = None,
        verbose: bool = False,
        rng: PRNGKey | None = None,
        dataset_statistics: Data | None = None,
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict. The only required key is "model", but other configuration
                may be saved for posterity.
            example_batch (Dict[str, Any]): Example batch.
            text_processor (Any, optional): Preprocessor for text inputs.
            verbose (bool, optional): Whether to print out a summary of the model.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        module = CrossFormerModule.create(**config["model"])
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        example_batch = multihost_utils.process_allgather(example_batch)
        example_batch = jax.tree_map(lambda x: x[:1], example_batch)

        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["timestep_pad_mask"],
        )

        if verbose:
            print(
                module.tabulate(rng, *init_args, train=False, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, and tokenizer details

        @jax.jit
        def _init(rng):
            return module.init(rng, *init_args, train=False)

        params = _init(rng)["params"]

        return cls(
            module=module,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )


def _verify_shapes(
    pytree,
    name: str,
    example_pytree,
    starting_dim: int = 0,
    strict: bool = False,
    raise_error: bool = True,
    silent: bool = False,
):
    weak_fail, fail = False, False
    pytree_flat = flax.traverse_util.flatten_dict(pytree)
    example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

    # Check that all elements are present
    if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
        if not silent:
            extra = set(pytree_flat.keys()) - set(example_pytree_flat.keys())
            if extra:
                logging.warning(
                    "'%s' contains extra items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in extra},
                )
            missing = set(example_pytree_flat.keys()) - set(pytree_flat.keys())
            if missing:
                logging.warning(
                    "'%s' is missing items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in missing},
                )
        weak_fail = True

    mismatched_keys = {
        k: f"{pytree_flat[k].shape} != {example_pytree_flat[k].shape}"
        for k in pytree_flat
        if k in example_pytree_flat
        and pytree_flat[k].shape[starting_dim:]
        != example_pytree_flat[k].shape[starting_dim:]
    }
    if mismatched_keys:
        if not silent:
            logging.error(
                "'%s' contains mismatched shapes compared to example_batch: %s",
                name,
                flax.core.pretty_repr(
                    {"/".join(k): v for k, v in mismatched_keys.items()}
                ),
            )
        fail = True

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError(f"{name} does not match example batch.")

    return weak_fail or fail


def _download_from_huggingface(huggingface_repo_id: str):
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder
