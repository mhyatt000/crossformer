"""Preflight for the eval callback pipeline: dummy batches, no model.

Seeds ``EvalContext.pred`` / ``pred_flow`` with the dummy ``act.base`` (an
"oracle" whose prediction equals ground truth; flow interpolates noise ->
target), then drives the real ``EvalLoop`` so hist/chunk/flow-pca/rast run
end-to-end and log to wandb.

Usage:
    uv run scripts/preflight/eval_smoke.py
    uv run scripts/preflight/eval_smoke.py --wandb.no-use   # skip wandb upload
    uv run scripts/preflight/eval_smoke.py --rast.every 0   # skip rast render
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from rich import print
import tyro

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.run.dummy import make_fake_batch
from crossformer.run.xflow_eval import EvalLoop
from crossformer.utils.callbacks.base import EvalContext
from crossformer.utils.callbacks.denorm import ActionBatchDenormalizer
from crossformer.utils.callbacks.hist import ChunkCallback, HistCallback
from crossformer.utils.callbacks.rast import RastCallback
from crossformer.utils.callbacks.viz import FlowPCACallback
from crossformer.utils.jax_utils import str2np

DS_NAME = "dummy_ds"


@dataclass
class Config:
    """Eval callback preflight config."""

    name: str = ""
    batch_size: int = 4
    obs_horizon: int = 1
    action_horizon: int = 20
    embodiment: tuple[str, ...] = ("single", "cart_gripper")
    flow_frames: int = 8  # synthetic flow steps (noise -> act.base)
    seed: int = 0

    hist: HistCallback = default(HistCallback(every=1))
    chunks: ChunkCallback = default(ChunkCallback(every=1))
    viz: FlowPCACallback = default(FlowPCACallback(every=1, fps=4))
    rast: RastCallback = default(RastCallback(every=1, eval_frames=8))
    wandb: cn.Wandb = default(cn.Wandb(project="crossformer-preflight", group="eval-smoke"))


def _names(*xs: str) -> np.ndarray:
    width = max(len(x) for x in xs)
    out = np.zeros((len(xs), width), dtype=np.uint8)
    for i, x in enumerate(xs):
        enc = str2np(x)
        out[i, : len(enc)] = enc
    return out


@dataclass
class FakeLoader:
    """Yields dummy bundled-action batches with dataset-name info."""

    cfg: Config
    n: int = 64

    def __iter__(self) -> Iterator[dict]:
        for _ in range(self.n):
            batch = make_fake_batch(
                batch_size=self.cfg.batch_size,
                obs_horizon=self.cfg.obs_horizon,
                action_horizon=self.cfg.action_horizon,
                embodiment=self.cfg.embodiment,
            )
            batch["info"] = {"dataset_name": _names(*[DS_NAME] * self.cfg.batch_size)}
            yield batch


@dataclass
class OracleEvalLoop(EvalLoop):
    """EvalLoop whose "model" is an oracle: pred == act.base.

    Pre-seeds the ctx prediction caches so no model forward ever runs;
    everything downstream (adapt, denorm, render, log) is the real path.
    """

    flow_frames: int = 8
    seed: int = 0
    _tick: int = field(default=0, init=False, repr=False)

    def _make_ctx(self, model: object, params: object, step: int) -> EvalContext:
        ctx = super()._make_ctx(model, params, step)
        base = np.asarray(ctx.batch["act"]["base"], dtype=np.float32)
        if base.ndim == 3:
            base = base[:, None]
        self._tick += 1
        rng = np.random.default_rng(self.seed + self._tick)
        noise = rng.standard_normal(base.shape).astype(np.float32)
        alphas = np.linspace(0.0, 1.0, self.flow_frames, dtype=np.float32).reshape(-1, 1, 1, 1, 1)
        ctx.__dict__["pred"] = base
        ctx.__dict__["pred_flow"] = (1.0 - alphas) * noise[None] + alphas * base[None]
        return ctx


def main(cfg: Config) -> None:
    if cfg.rast.every > 0 and (cfg.rast.urdf is None or not Path(cfg.rast.urdf).exists()):
        print(f"[yellow]rast disabled: urdf {cfg.rast.urdf} not found[/]")
        cfg.rast.every = 0
    if cfg.rast.every > 0:
        cfg.rast.cams = tuple(p for p in cfg.rast.cams if Path(p).exists())

    run = cfg.wandb.initialize(cfg)

    sent: dict = {}

    def log_and_record(metrics: dict, step: int | None = None) -> None:
        sent.update(metrics)
        cfg.wandb.log(metrics, step=step)

    loop = OracleEvalLoop(
        loader=FakeLoader(cfg),
        callbacks=[cfg.hist, cfg.chunks, cfg.viz, cfg.rast],
        denorm=ActionBatchDenormalizer({DS_NAME: {"action": {}}}),
        wandb_log=log_and_record,
        flow_frames=cfg.flow_frames,
        seed=cfg.seed,
    )
    loop(model=None, params=None, step=0, is_last=True)

    expected = {cb.name for cb in loop.callbacks if cb.every > 0}
    logged = {k.split("/", 1)[0] for k in sent}
    missing = expected - logged
    assert not missing, f"callbacks logged nothing: {missing}"

    print({"ok": True, "logged_keys": sorted(sent)})
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
