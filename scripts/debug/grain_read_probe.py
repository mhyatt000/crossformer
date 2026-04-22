from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
import multiprocessing
import os
import time
from types import SimpleNamespace
from typing import Any

from absl import flags
from rich import print
from tqdm import tqdm

if multiprocessing.parent_process() is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    setattr(flags.FLAGS, "jax_allow_unused_gpus", True)

import grain
import jax
import numpy as np
from PIL import Image
from rich.rule import Rule
from rich.table import Table
import tyro

from crossformer.cn.dataset.mix import Arec, DataSource
from crossformer.cn.dataset.transform.traj import TrajectoryTransform
from crossformer.utils.peace_and_quiet import (
    install_absl_filter,
    patch_arec_source,
    quiet_jax_xla_bridge,
)


@dataclass
class GrainTransformCfg:
    traj: TrajectoryTransform = field(default_factory=lambda: TrajectoryTransform(name=""))
    skip_norm_keys: tuple[str, ...] = ("proprio_bimanual", "proprio_mano")


@dataclass
class GrainDataCfg:
    mix: Any
    loader: Any
    transform: GrainTransformCfg = field(default_factory=GrainTransformCfg)
    recompute: bool = False

    @property
    def traj(self):
        return self.transform.traj


@dataclass
class GrainTrainLike:
    mix: str = "xgym_sweep"
    batch_size: int = 4096
    seed: int = 42
    recompute: bool = False
    window_size: int = 1
    loader: Any = None
    transform: GrainTransformCfg = field(default_factory=GrainTransformCfg)
    data: GrainDataCfg = field(init=False)

    def __post_init__(self):
        loader = self.loader or _make_default_loader()
        mix = SimpleNamespace(value=DataSource.REGISTRY[self.mix])
        loader = replace(loader, global_batch_size=self.batch_size)
        transform = replace(
            self.transform,
            traj=replace(self.transform.traj, window_size=self.window_size),
        )
        self.data = GrainDataCfg(
            mix=mix,
            loader=loader,
            transform=transform,
            recompute=self.recompute,
        )


@dataclass
class Config(GrainTrainLike):
    lightweight_cfg: bool = False
    batches: int = 10
    mp: int = 0
    shuffle: bool = True
    resize: tuple[int, int] = (64, 64)
    mp_prefetch: int = 1
    read_threads: int = 48
    read_prefetch: int = 64


def _spec(x, *, simple: bool = False):
    from crossformer.utils.spec import spec

    return spec(x, simple=simple)


def _make_default_loader():
    from crossformer.cn.dataset.dataset import Loader

    return Loader(use_grain=True)


def _make_train_cfg_old(
    mix: str,
    batch_size: int,
    loader: Any,
    recompute: bool,
    seed: int,
    window_size: int,
):
    import crossformer.cn as cn
    from crossformer.cn.dataset import DataSourceE

    return cn.Train(
        data=cn.Dataset(
            mix=DataSourceE[mix],
            loader=replace(loader, global_batch_size=batch_size),
            recompute=recompute,
        ),
        seed=seed,
        window_size=window_size,
        verbosity=0,
    )


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label}: {dt:.3f}s")
    return out, dt


def _resize_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    return np.asarray(Image.fromarray(img).resize((w, h), resample=Image.BILINEAR), dtype=img.dtype)


def _early_resize_sample(x: dict, size: tuple[int, int]) -> dict:
    image = x.get("observation", {}).get("image")
    if not isinstance(image, dict):
        return x
    x["observation"]["image"] = {k: _resize_image(v, size) for k, v in image.items()}
    return x


def _replace_images_with_dummy(x: dict, size: tuple[int, int]) -> dict:
    image = x.get("observation", {}).get("image")
    if not isinstance(image, dict):
        return x
    h, w = size
    x["observation"]["image"] = {k: np.zeros((h, w, v.shape[-1]), dtype=v.dtype) for k, v in image.items()}
    return x


def _worker_cpu_only_init(worker_index: int, worker_count: int) -> None:
    del worker_index, worker_count
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    install_absl_filter()
    with quiet_jax_xla_bridge():
        _ = jax.devices("cpu")


def build_dataset(cfg: Config):
    from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory, make_source_by_mix

    patch_arec_source()

    loader = cfg.loader or _make_default_loader()
    train_cfg = (
        cfg
        if cfg.lightweight_cfg
        else _make_train_cfg_old(
            mix=cfg.mix,
            batch_size=cfg.batch_size,
            loader=loader,
            recompute=cfg.recompute,
            seed=cfg.seed,
            window_size=cfg.window_size,
        )
    )
    factory = GrainDataFactory(mp=cfg.mp, shuffle=cfg.shuffle, resize=cfg.resize)

    # return iter(factory.make(train_cfg).dataset)
    _apply_fd_limit(512**2)

    mix = train_cfg.data.mix.value.flatten()
    mix = [Arec.from_name(name) for name, _weight in mix]
    sources = [make_source_by_mix(m, train_cfg) for m in mix]
    max_a = max(m.embodiment.action_dim for m in mix)

    def build_sources():
        if len(sources) == 1:
            _, dconfig = sources[0]
            return factory.source2ds(dconfig, train_cfg, dataset=mix[0], max_a=max_a), sources[0][1]
        dsets = [factory.source2ds(dc, train_cfg, dataset=m, max_a=max_a) for (_, dc), m in zip(sources, mix)]
        return factory.pad_and_mix(dsets), sources[0][1]

    ds = _timed("source2ds", build_sources)[0][0]
    ds, _ = _timed("early resize map", lambda: ds.map(partial(_early_resize_sample, size=cfg.resize)))
    ds, _ = _timed("dummy image map", lambda: ds.map(partial(_replace_images_with_dummy, size=cfg.resize)))

    def build_iter():
        _ds = ds.seed(train_cfg.seed)
        if factory.shuffle:
            _ds = _ds.shuffle()
        _ds = _ds.repeat()
        return (
            _ds.map(lambda x: jax.tree.map(lambda y: np.array(y), x))
            .to_iter_dataset(
                grain.ReadOptions(
                    num_threads=cfg.read_threads,
                    prefetch_buffer_size=cfg.read_prefetch,
                )
            )
            .batch(train_cfg.data.loader.batch_size, drop_remainder=True)
        )

    it_ds, _ = _timed("to_iter_dataset+batch", build_iter)
    if cfg.mp > 0:
        it_ds = it_ds.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=cfg.mp, per_worker_buffer_size=cfg.mp_prefetch),
            worker_init_fn=_worker_cpu_only_init,
        )
    return it_ds


def main(cfg: Config) -> None:
    print(cfg)
    install_absl_filter()
    print(Rule("grain read probe", style="bold cyan"))
    print(
        {
            "mix": cfg.mix,
            "batch_size": cfg.batch_size,
            "lightweight_cfg": cfg.lightweight_cfg,
            "mp": cfg.mp,
            "mp_prefetch": cfg.mp_prefetch,
            "read_threads": cfg.read_threads,
            "read_prefetch": cfg.read_prefetch,
            "resize": cfg.resize,
        }
    )

    dataset = build_dataset(cfg)
    dsit = iter(dataset)

    times = []
    table = Table(title="batch timings")
    table.add_column("batch")
    table.add_column("seconds")

    for i in tqdm(range(cfg.batches)):
        t0 = time.perf_counter()
        batch = next(dsit)
        dt = time.perf_counter() - t0
        times.append(dt)
        table.add_row(str(i), f"{dt:.3f}")
        if i == 0:
            print(_spec(batch, simple=True))

    print(table)
    print(
        {
            "first_batch_s": round(times[0], 3),
            "mean_batch_s": round(sum(times) / len(times), 3),
            "min_batch_s": round(min(times), 3),
            "max_batch_s": round(max(times), 3),
        }
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
