from __future__ import annotations

import logging
import time

import grain
from grain._src.python import dataset as gd
import grain.experimental as ge
from rich.pretty import pprint
from tqdm import tqdm
import tyro

from crossformer.data.grain import pipelines
from crossformer.data.grain.map.window import mk_chunk, WindowFnIterDataset
from crossformer.utils.spec import spec
from scripts.debug import data_grain

log = logging.getLogger(__name__)


# Define a dummy slow preprocessing function
def _dummy_slow_fn(x):
    time.sleep(0.1)
    return x


def main(cfg: data_grain.Config) -> None:
    grain.config.update("py_debug_mode", True)
    dummy = pipelines.dummy_data
    use_dummy = True

    source = grain.MapDataset.range(1000)
    source = source.map(dummy) if use_dummy else source

    def add_info(x):
        x["info"] = {"id": {"episode_id": 0, "step_id": 0}}
        return x

    source = source.map(add_info) if use_dummy else source

    options = ge.DatasetOptions(
        execution_tracking_mode=ge.ExecutionTrackingMode.STAGE_TIMING,
    )

    source = source.map(_dummy_slow_fn)
    ds = source.to_iter_dataset()

    a, o = 50, 1
    maxao = max(a, o)
    noop = lambda x: x
    window_fn = mk_chunk(a, o) if use_dummy else noop
    pprint((use_dummy, window_fn))
    ds = WindowFnIterDataset(ds, window_size=maxao, window_fn=window_fn)

    performance = ge.pick_performance_config(
        ds,
        ram_budget_mb=1024 * 1024,
        max_workers=None,
        max_buffer_size=4,
    )
    pprint(performance)

    ds = ds.map(_dummy_slow_fn)
    # put this at the very end to enable tracking
    # we must have logging enabled to at least INFO level
    # also ensure grain.config.update("py_debug_mode", True) is set
    ds = ge.WithOptionsIterDataset(ds, options)
    dsit = iter(ds)

    if use_dummy:
        pprint(spec(next(iter(source))["action"]))
        pprint(spec(next(iter(source))["observation"]))
        pprint(spec(next(dsit)["action"]))

    for _ in tqdm(range(10)):
        e = next(dsit)
        # pprint((e[0],e[-1]))

    # stats = dsit._initialize_stats(options.execution_tracking_mode)
    summary = gd.dataset.get_execution_summary(dsit)  # must run on iterator
    print(gd.stats.pretty_format_summary(summary))


if __name__ == "__main__":
    main(tyro.cli(data_grain.Config))
