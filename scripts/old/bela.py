from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os

# from mpi4py import MPI
import socket

from finetune import create_optimizer, TrainState
import flax
from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
import jax
from jax.debug import visualize_array_sharding as vas
import jax.distributed
from jax.experimental import multihost_utils as mx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import ConfigDict
import numpy as np
import optax
from rich.pretty import pprint as _pprint
from tqdm import tqdm
import tyro

from crossformer import cn
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)
from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.model.crossformer_module import (
    CrossFormerModule,
    CrossFormerTransformer,
)
from crossformer.utils.train_utils import check_config_diff, merge_params, process_text

pprint = lambda *args, **kwargs: (_pprint(*args, **kwargs), print())


def make_dataset(cfg, model, mesh):
    def shard(batch):
        return mx.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    def process_batch(batch):
        batch = process_text(batch, model.text_processor)
        del batch["dataset_name"]
        return batch

    tfdataset = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS, train=True)
    data = tfdataset.iterator(prefetch=cfg.data.loader.prefetch)
    data = map(shard, map(process_batch, data))

    example_batch = next(data)
    spec = lambda _x: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), _x)
    pprint(spec(example_batch))
    return tfdataset, data, example_batch


def loss_fn_new(module: nnx.Module, batch, train=True):
    embeds = module.fwd_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )

    loss, metrics = 0, {}
    for name, head in module.heads.items():
        l, m = head.loss(
            embeds,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            action_head_mask=batch["action_head_masks"][name],
            train=train,
        )

        # weight loss by number of samples from each head
        frac = (batch["action_head_masks"][name].sum()) / len(batch["action"])
        loss += l * frac * head.loss_weight
        metrics[name] = m

    metrics["total_loss"] = loss
    return loss, metrics


class CrossFormerNNX(nnx.Module):
    def __init__(self, transformer: CrossFormerTransformer, heads: dict[str, nn.Module], rngs):
        self.transformer = bridge.ToNNX(transformer, rngs=rngs)
        self.heads = {k: bridge.ToNNX(h, rngs=rngs) for k, h in heads.items()}

    def __call__(self, observations, tasks, timestep_pad_mask, train=True, verbose=False):
        outs = self.fwd_transformer(observations, tasks, timestep_pad_mask, train, verbose)
        head_outputs = self.fwd_head(outs, train)
        return outs, head_outputs

    def fwd_transformer(self, observations, tasks, timestep_pad_mask, train=True, verbose=False):
        outs = self.transformer(observations, tasks, timestep_pad_mask, train=train, verbose=verbose)
        return outs

    def fwd_head(self, outs, train=True):
        head_outputs = {}
        for head_name, head in self.heads.items():
            head_outputs[head_name] = head(outs, train=train)
        return head_outputs

    # def loss_head():


def loss_fn(module: nnx.Module, batch, train=True):
    embeds = module.crossformer_transformer(
        batch["observation"],
        batch["task"],
        batch["observation"]["timestep_pad_mask"],
        train=train,
    )

    loss, metrics = 0, {}
    for name, head in module.heads.items():
        l, m = head.loss(
            embeds,  # action head knows to pull out the "action" readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            action_head_mask=batch["action_head_masks"][name],
            train=train,
        )

        # weight loss by number of samples from each head
        frac = (batch["action_head_masks"][name].sum()) / len(batch["action"])
        loss += l * frac * head.loss_weight
        metrics[name] = m

    metrics["total_loss"] = loss
    return loss, metrics


@nnx.jit
def train_step(module: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""

    grad_fn = nnx.value_and_grad(loss_fn_new, has_aux=True)
    (loss, info), grads = grad_fn(module, batch, train=True)
    metrics.update(loss=loss)

    info.update(
        {
            "grad_norm": optax.global_norm(grads),
            # "update_norm": optax.global_norm(updates),
            "param_norm": param_norm_callable(module.params),
            "learning_rate": lr_callable(state.step),
        }
    )

    optimizer.update(grads)


@nnx.jit
def pred_step(model: nnx.Module, batch):
    logits = model(batch["image"])
    return logits.argmax(axis=1)


@nnx.jit
def eval_step(model: nnx.Module, metrics: nnx.MultiMetric, batch):
    loss, _logits = loss_fn(model, batch)
    metrics.update(loss=loss)


def run_nnx(module, optimizer, metrics, data):
    # nsteps = train_ds.cardinality().numpy() // num_epochs
    nsteps = 5_000_000
    every = 1_000

    hist = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    nnx.reseed(module, dropout=0)
    module.train()
    for i in tqdm(range(nsteps)):
        batch = next(data)
        train_step(module, optimizer, metrics, batch)

        if (i + 1) % every == 0:  # one training epoch has passed
            print("quitting")
            quit()
            continue

            # Log training metrics
            for metric, value in metrics.compute().items():  # compute metrics
                hist[f"train_{metric}"].append(value)  # record metrics
            metrics.reset()  # reset metrics for test set

            # Compute metrics on the test set after each training epoch
            # for test_batch in test_ds.as_numpy_iterator():
            # module.eval()
            # eval_step(module, metrics, test_batch)
            # module.train()

            # Log test metrics
            for metric, value in metrics.compute().items():
                hist[f"test_{metric}"].append(value)
            metrics.reset()  # reset metrics for next training epoch

            print(f"loss: {hist['train_loss'][-1]}, accuracy: {hist['train_accuracy'][-1] * 100}")
            print(f"loss: {hist['test_loss'][-1]}, accuracy: {hist['test_accuracy'][-1] * 100}")


def run_linen(train_state, model, data, cfg, param_norm_callable, lr_callable, shards):
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
            head_sample_fraction = (batch["action_head_masks"][head_name].sum()) / len(batch["action"])
            action_loss += head_loss * head_sample_fraction * head.loss_weight
            action_metrics[head_name] = head_metrics
        action_metrics["total_loss"] = action_loss

        return action_loss, action_metrics

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=(shards["rep"], shards["ddp"]),
        out_shardings=shards["rep"],
    )
    def train_step(state: TrainState, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (_loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
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

    for i in tqdm(range(cfg.steps), total=cfg.steps, dynamic_ncols=True):
        batch = next(data)

        train_state, update_info = train_step(train_state, batch)

        if i % 100 == 0:
            spec = lambda x: jax.tree.map(lambda y: y.shape, x)
            pprint(spec(batch))
            pprint(update_info)
            try:
                a = batch["action"]
                vas(a.reshape(a.shape[0], -1))
            except Exception as e:
                pass


@dataclass
class Shmodel:
    """A sharded model."""

    mesh: Mesh

    def create(self, cls, *args, **kwargs):
        @nnx.jit
        def create_sharded_model():
            # Unsharded at this moment.
            _model = cls(*args, **kwargs)
            state = nnx.state(_model)  # The model's state, a pure pytree.
            # Strip out the annotations from state.
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(_model, sharded_state)  # The model is sharded now!
            return _model

        with self.mesh:
            shmodel = create_sharded_model()
        return shmodel


def show_shardings(shards: dict[str, NamedSharding]):
    for k, shard in shards.items():
        pprint({"name": k, "mesh": shard})
        x = jax.random.normal(jax.random.PRNGKey(42), (jax.device_count(), jax.device_count()))
        _x = jax.device_put(x, shard)
        vas(_x)


def make_shards():
    pprint(jax.devices())
    mesh = Mesh(
        devices=np.array(jax.devices()).reshape(-1, 1),
        axis_names=("batch", "model"),
    )
    pprint(mesh)

    shards = {
        "ddp": NamedSharding(mesh, PartitionSpec("batch", None)),
        "rep": NamedSharding(mesh, PartitionSpec()),
    }
    show_shardings(shards)
    return mesh, shards


def main(cfg: cn.Train):
    pprint(cfg)

    #
    # init distributed devices
    #

    for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(var, None)
    # os.environ["NCCL_SOCKET_IFNAME"] = "hsn"

    host = socket.gethostname()
    jax.distributed.initialize(
        coordinator_address=f"{host}:29500",
        num_processes=1,  # len(nodes),
        process_id=0,  # id,
        # cluster_detection_method='mpi4py'
    )

    mesh, shards = make_shards()
    DDP, REP = shards["ddp"], shards["rep"]

    #
    # init model
    #

    ptm = CrossFormerModel.load_pretrained("hf://rail-berkeley/crossformer", None)

    flat_config = flax.traverse_util.flatten_dict(ptm.config, keep_empty_nodes=True)
    flat_config = cfg.model.delete(flat_config)

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(cfg.model.create())
    config = config.to_dict()
    check_config_diff(config, ptm.config)

    # model = Shmodel(mesh).create(FeedForward, features=1024, hidden_dim=32, rngs=nnx.Rngs(0))

    spec = lambda x: jax.tree.map(lambda y: y.shape, x)
    # pprint(spec(model.example_batch))

    # trigger compilation
    # ptm.module.tabulate( rng, *batch, depth=2)

    #
    # init data
    #

    tfdataset, data, example_batch = make_dataset(cfg, ptm, mesh)
    init_batch = (
        example_batch["observation"],
        example_batch["task"],
        example_batch["observation"]["timestep_pad_mask"],
        False,  # train
        False,  # verbose
    )

    #
    # fix model to cfg
    #

    rng = jax.random.PRNGKey(cfg.seed)
    rng, init_rng = jax.random.split(rng)
    model = CrossFormerModel.from_config(
        config,
        example_batch,
        ptm.text_processor,
        rng=init_rng,
        dataset_statistics=tfdataset.dataset_statistics,
        verbose=True,
    )

    merged_params = merge_params(model.params, ptm.params)
    model = model.replace(params=merged_params)
    try:
        del ptm
    except Exception as e:
        print(f"Failed to delete ptm: {e}")

    use_nnx = False
    use_linen = not use_nnx

    if use_nnx:
        module: CrossFormerModule = model.module
        module = CrossFormerNNX(module.crossformer_transformer, module.heads, rngs=nnx.Rngs(0))
        bridge.lazy_init(module, *init_batch)
        y = module(*init_batch)
        pprint(spec(y))

        optimizer = nnx.Optimizer(module, optax.adamw(0.005, 0.9))
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

    if use_linen:
        tx, lr_callable, param_norm_callable = create_optimizer(model.params, **cfg.optimizer.create())
        train_state = TrainState.create(model=model, tx=tx, rng=rng)

    #
    # run
    #

    with mesh:
        if use_nnx:
            run_nnx(module, optimizer, metrics, data)
        if use_linen:
            run_linen(train_state, model, data, cfg, param_norm_callable, lr_callable, shards)


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
