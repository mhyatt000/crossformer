import jax  # noqa
from rich.pretty import pprint
import tyro

from crossformer import cn
from crossformer.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS

from tqdm import tqdm


def main(cfg: cn.Train):
    dataset = cfg.data.create(OXE_STANDARDIZATION_TRANSFORMS, train=True)
    data_iter = dataset.iterator(prefetch=cfg.data.loader.prefetch)
    # data_iter = map(shard, map(process_batch, data_iter))

    example_batch = batch = next(data_iter)
    spec = lambda _x: jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), _x)

    pprint(spec(batch))
    # pprint(batch['action'].mean((-1,-2)))
    # pprint(batch['action_pad_mask'].mean((-1,-2, -3)))
    # pprint(batch['action_head_masks']['single_arm'].mean(-1))

    act_total = 0
    for batch in tqdm(data_iter):
        pass
        # act = batch["action"]
        # total = act.sum((0, 1, 2))[-1]
        # pprint((total, bool(total)))


if __name__ == "__main__":
    main(tyro.cli(cn.Train))
