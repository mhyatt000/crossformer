# for Flow Adjustments
from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import numpy as np
from rich.pretty import pprint
import tyro

from crossformer import cn
from crossformer.data.grain.map.window import FlatMapDataset, MyFlatMap, WindowedFlatMap, WindowFlatDataset  # noqa
from crossformer.model.components.action_heads import AdjFlowHead
from crossformer.model.components.base import TokenGroup

log = logging.getLogger(__name__)


@dataclass
class Config(cn.Train):
    arec_path: Path | None = None
    dataset_name: str | None = None
    recompute: bool = False  # recompute data stats? y/n
    once: bool = True


@dataclass
class ArecReader:
    # reader
    path: Path


def main(cfg: Config) -> None:
    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # @deprecate('let grain do it', strict=False)
    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    myspec = {
        "action": (1, 50, 8),
        "action_head_masks": {
            "mano": (),
            "single_arm": (),
        },
        "action_pad_mask": (1, 50, 8),
        "observation": {
            "image": {
                "overhead": (1, 224, 224, 3),
                "side": (1, 224, 224, 3),
                "worm": (1, 224, 224, 3),
                "wrist": (1, 224, 224, 3),
            },
            "pad_mask_dict": {
                "image": {
                    "overhead": (1,),
                    "side": (1,),
                    "worm": (1,),
                    "wrist": (1,),
                },
                "proprio": {
                    "gripper": (1,),
                    "joints": (1,),
                    "position": (1,),
                    "single_arm": (1,),
                },
                "timestep": (1,),
            },
            "proprio": {
                "gripper": (1, 1),
                "joints": (1, 7),
                "position": (1, 6),
                "single_arm": (1, 14),
            },
            "task_completed": (1, 50),
            "timestep": (1,),
            "timestep_pad_mask": (1,),
        },
        "task": {
            "image": {
                "overhead": (224, 224, 3),
                "side": (224, 224, 3),
                "worm": (224, 224, 3),
                "wrist": (224, 224, 3),
            },
            "language.embedding": (512,),
            "pad_mask_dict": {
                "image": {
                    "overhead": (),
                    "side": (),
                    "worm": (),
                    "wrist": (),
                },
                "language.embedding": (),
                "proprio": {
                    "gripper": (),
                    "joints": (),
                    "position": (),
                    "single_arm": (),
                },
                "timestep": (),
            },
            "proprio": {
                "gripper": (1,),
                "joints": (7,),
                "position": (6,),
                "single_arm": (14,),
            },
            "timestep": (),
        },
    }

    isleaf = lambda y: not isinstance(y, dict)
    key = jax.random.key(0)
    exbatch = jax.tree.map(lambda x: jax.random.normal(key, x, dtype=np.float32), myspec, is_leaf=isleaf)
    embeddings = {"action_single_arm": jax.random.normal(key, (1, 50, 512))}

    # unsqueeze
    unsqueeze = lambda x: jnp.stack([x] * 4, axis=0)
    exbatch = jax.tree.map(unsqueeze, exbatch)
    embeddings = jax.tree.map(unsqueeze, embeddings)
    embeddings = {k: TokenGroup.create(v) for k, v in embeddings.items()}

    # pprint(spec(exbatch, simple=True))

    head = AdjFlowHead(
        readout_key="action_single_arm",
        pool_strategy="mean",
        action_horizon=50,  # this is important for adjflow
        action_dim=8,  # this is important for adjflow
        max_action=5,
        loss_type="mse",
        num_preds=0,
        flow_steps=10,
    )

    pprint(head)
    weights = head.init(
        jax.random.key(0),
        transformer_outputs=embeddings,
        # create from action
        time=jax.random.uniform(jax.random.key(1), (*exbatch["action"].shape[:-1], 1)),
        current=exbatch["action"],
        train=True,
    )
    bound = head.bind(weights, rngs={"dropout": jax.random.key(2)})

    l, metrics = bound.loss(embeddings=embeddings, batch=exbatch, train=True)
    pprint((l, metrics))


if __name__ == "__main__":
    main(tyro.cli(Config))
