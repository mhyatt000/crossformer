"""ensure grain data looks like tfds data"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from rich.pretty import pprint
import tyro

from crossformer import cn
from crossformer.data.grain import pipelines
from crossformer.data.grain.map.window import FlatMapDataset, WindowedFlatMap, WindowFlatDataset  # noqa
from crossformer.data.grain.utils import flat
from crossformer.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from crossformer.utils.spec import diff, spec

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

    _ds, dataset_config, tfconfig = pipelines.make_data_source(cfg)
    gd = pipelines.make_single_dataset(
        dataset_config,
        shard_fn=shard,
        train=True,
        tfconfig=tfconfig,
        shuffle_buffer_size=1,
        seed=cfg.seed,
    )

    pprint(gd)
    gb = next(iter(gd.dataset))
    pprint(spec(gb))

    tfd = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS, train=True)
    tfd = tfd.iterator(prefetch=cfg.data.loader.prefetch)
    tfb = batch = next(tfd)

    pprint(
        spec(
            diff(
                spec(flat(tfb)),
                spec(flat(gb)),
            )
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
