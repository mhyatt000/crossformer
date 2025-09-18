from crossformer.cn import ModelFactory
from crossformer import cn
from crossformer.cn.dataset import Dataset, Reader, DataSourceE  # , TransformE
import jax
from crossformer.utils.spec import ModuleSpec
from rich.pretty import pprint
from crossformer.data.oxe import make_oxe_dataset_kwargs_and_weights
from crossformer.data.dataset import make_interleaved_dataset, make_single_dataset
from crossformer.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
)
import xgym


import tyro

def main(cfg: ModelFactory):
    pprint(cfg)
    pprint(cfg.create())

def main(cfg: Dataset):
    pprint(cfg)

    dataset = cfg.create(OXE_STANDARDIZATION_TRANSFORMS)

    spec = lambda _x: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), _x)
    batch = next(iter(dataset))
    pprint(spec(batch))

def main(cfg: cn.Train):
    pprint(cfg)
    dataset = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS)

    spec = lambda _x: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), _x)
    batch = next(iter(dataset))
    pprint(spec(batch))

if __name__ == "__main__":
    # main(tyro.cli(ModelFactory))
    # main(tyro.cli(Dataset))
    main(tyro.cli(cn.Train))
