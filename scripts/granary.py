from __future__ import annotations

from arec import ArrayRecordBuilder, unpack_record
import grain
from grain._src.python import options as grain_options
import jax
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm


class ResizeAndCrop(grain.transforms.Map):
    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = element["image"]
        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)
        image = image[:224, :224]
        element["image"] = image
        return element


transformations = [ResizeAndCrop()]

builder = ArrayRecordBuilder(
    name="tmp_spec",
    root="~/.cache/arecs",
    version="v1",  # bump when schema/layout changes
    shard_size=1000,  # records per shard
    writer_options="group_size:1",  # passed directly to ArrayRecordWriter
)
len(builder)

store = builder._ds.__getitems__(list(range(len(builder._ds))))
store = [unpack_record(x) for x in store]

# for x in tqdm(store):
# unpack_record(x)


def spec(x):
    return jax.tree.map(lambda arr: (arr.shape, str(arr.dtype)), x)


h = 4  # horizon
batch_size = 256
read_options = grain_options.ReadOptions(num_threads=10, prefetch_buffer_size=8 * batch_size)

ds = (
    # You can also use a shortcut grain.MapDataset.range for
    # range-like input.
    grain.MapDataset.source(store)
    # .map(unpack_record)
    .batch(batch_size=h)
    .shuffle(seed=10)  # Shuffles globally.
    .to_iter_dataset()
    # .repeat()
    # .filter(lambda e: isinstance(e, dict))
    # .to_iter_dataset(read_options=read_options)
    # .map(lambda x: x )  # Maps each element.
    .batch(batch_size=batch_size)  # Batches consecutive elements.
)

# ds = grain.experimental.ThreadPrefetchIterDataset(ds, prefetch_buffer_size=cpu_buffer_size)

# ds = ds.map(jax.device_put)

# ds = grain.MapDataset.mix([ds1, ds2], weights=[0.7, 0.3])

""" AUTOTUNE
performance_config = grain.experimental.pick_performance_config(
        ds=ds,
        ram_budget_mb=1024,
        max_workers=None,
        max_buffer_size=None
    )
ds = ds.to_iter_dataset(read_options=performance_config.read_options)
"""

# ds = ds.mp_prefetch( grain.multiprocessing.MultiprocessingOptions(num_workers=3, per_worker_buffer_size=1000))


# pprint(ds[0])
# pprint(list(ds[:5]))

for i, x in tqdm(enumerate(ds), total=len(store) / (256 * 4)):
    pprint(spec(x))
    # if i >= 10:
    # break

# jax.profiler.stop_trace()
