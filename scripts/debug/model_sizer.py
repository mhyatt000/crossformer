"""Sweep (model size x batch size x image size) feasibility on a real data mix.

For each combo this builds the full xflow training setup (GrainDataFactory ->
CrossFormerModel -> optimizer -> train_step) and measures params, compile time,
step time, throughput, and peak device memory. Each combo runs in a fresh
subprocess so an OOM cannot poison the sweep and GPU memory is fully released
between combos; larger batches at a (size, image) that already OOM'd are
skipped automatically.

Usage:
    uv run scripts/debug/model_sizer.py --mix xgym_lift
    uv run scripts/debug/model_sizer.py --sizes vanilla vit_s detr --batch-sizes 128 256 512
    uv run scripts/debug/model_sizer.py --image-sizes 64x64 480x640 --dry
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import tyro

# -- config -------------------------------------------------------------------


@dataclass
class Config:
    """Model sizer sweep config."""

    mix: str = "xgym_lift"  # dataset mix (both embodiments for xgym_lift)
    sizes: tuple[str, ...] = ("vanilla", "vit_t", "vit_s", "detr")  # transformer sizes to sweep
    batch_sizes: tuple[int, ...] = (16, 128, 256, 512)  # global batch sizes (ascending for OOM pruning)
    image_sizes: tuple[str, ...] = ("64x64", "224x224", "480x640")  # HxW resize targets
    horizon: int = 50  # action horizon (match the run under investigation)
    steps: int = 8  # steps per combo (step 0 = compile, rest timed)
    lr: float = 1e-4  # lr (irrelevant to sizing, needed to build the optimizer)
    mp: int = 4  # grain multiproc workers
    timeout: int = 1800  # per-combo subprocess timeout (s)
    out: str = "model_sizer_results.json"  # results file (parent mode)
    dry: bool = False  # list combos without running
    # child mode: run exactly one combo (first element of each sweep list) and
    # dump a JSON result to this path. Set by the parent; not for manual use.
    child_result: str | None = None


def _parse_hw(s: str) -> tuple[int, int]:
    h, w = s.lower().split("x")
    return int(h), int(w)


# -- child: run one combo -----------------------------------------------------


def _count_params(params: Any) -> tuple[int, int]:
    """Return (total, frozen) param counts; frozen = anything under a "tips" subtree."""
    import flax
    import jax

    flat = flax.traverse_util.flatten_dict(params)
    total = sum(int(v.size) for v in flat.values())
    frozen = sum(int(v.size) for k, v in flat.items() if "tips" in k)
    _ = jax  # keep import local to child
    return total, frozen


def _peak_memory_gib() -> float:
    import jax

    peaks = []
    for d in jax.local_devices():
        stats = d.memory_stats() or {}
        peaks.append(stats.get("peak_bytes_in_use", 0))
    return max(peaks, default=0) / 2**30


def run_one(cfg: Config) -> dict[str, Any]:
    """Build data + model + train_step for one combo and measure it."""
    size, batch_size = cfg.sizes[0], cfg.batch_sizes[0]
    resize = _parse_hw(cfg.image_sizes[0])
    result: dict[str, Any] = {
        "size": size,
        "batch_size": batch_size,
        "image": cfg.image_sizes[0],
        "status": "ok",
    }

    import jax
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, PartitionSpec

    import crossformer.cn as cn
    from crossformer.cn.dataset import DataSourceE
    from crossformer.cn.dataset.dataset import Loader
    from crossformer.cn.model_factory import Size, Vision
    from crossformer.data.grain.loader import _apply_fd_limit, GrainDataFactory
    from crossformer.model.crossformer_model import CrossFormerModel
    from crossformer.run.train_step import make_train_step
    from crossformer.utils.callbacks.base import extract_bundled_actions, flatten_obs
    from crossformer.utils.jax_utils import initialize_compilation_cache
    from crossformer.utils.train_utils import create_optimizer, TrainState

    initialize_compilation_cache()
    devices = jax.devices()
    mesh = Mesh(devices, axis_names="batch")
    if batch_size % len(devices) != 0:
        batch_size = max(len(devices), (batch_size // len(devices)) * len(devices))
        result["batch_size"] = batch_size

    def shard(batch: Any) -> Any:
        return multihost_utils.host_local_array_to_global_array(batch, mesh, PartitionSpec("batch"))

    # Data
    _apply_fd_limit(512**2)
    train_cfg = cn.Train(
        data=cn.Dataset(mix=DataSourceE[cfg.mix], loader=Loader(use_grain=True, global_batch_size=batch_size)),
        seed=42,
        verbosity=0,
    )
    t0 = time.perf_counter()
    dataset = GrainDataFactory(mp=cfg.mp, resize=resize).make(train_cfg, shard_fn=shard, train=True)
    dsit = iter(dataset.dataset)
    batch = next(dsit)
    result["data_s"] = round(time.perf_counter() - t0, 1)

    obs_shape = batch["observation"]["image"].shape  # (B, W, V, H, W, C)
    result["obs_image_shape"] = list(obs_shape)
    max_a = int(batch["act"]["id"].shape[-1])
    max_w = int(batch["observation"]["timestep_pad_mask"].shape[1])
    result["max_a"], result["window"] = max_a, max_w

    # Model: same vision defaults as scripts/train/xflow.py, sweep the trunk size
    factory = cn.ModelFactory(
        size=Size(size),
        window=max_w,
        image_keys=(),
        proprio_keys=(),
        vision=Vision(stacked=True, stacked_encoder="tips", tips_variant="tips_v2_b14", stacked_freeze=True),
    )
    factory.xflow.max_dofs = max_a
    factory.xflow.max_horizon = cfg.horizon

    obs = flatten_obs(
        batch["observation"],
        (),
        view_mask=batch.get("mask", {}).get("view"),
        state=batch.get("state"),
        mask=batch.get("mask"),
    )
    task = batch.get("task", {"pad_mask_dict": {}})
    model_cfg = factory.create()
    model_cfg["optimizer"] = {
        "learning_rate": cfg.lr,
        "weight_decay": 1e-4,
        "clip_gradient": 1.0,
        "frozen_keys": ["*tips*"],  # matches xflow.py stacked_freeze wiring
    }

    t0 = time.perf_counter()
    model = CrossFormerModel.from_config(
        model_cfg,
        {"observation": dict(obs), "task": task},
        text_processor=None,
        rng=jax.random.PRNGKey(42),
        dataset_statistics=dataset.dataset_statistics,
    )
    result["init_s"] = round(time.perf_counter() - t0, 1)
    # NOTE: pretrained TIPS weights are NOT loaded — values don't affect
    # memory/speed, and skipping the load keeps the sweep fast.
    total, frozen = _count_params(model.params)
    result["n_params"] = total
    result["n_trainable"] = total - frozen

    tx, lr_callable, param_norm_callable = create_optimizer(model.params, **model_cfg["optimizer"])
    state = TrainState.create(model=model, tx=tx, rng=jax.random.PRNGKey(0))
    train_step = make_train_step(model.module, lr_callable, param_norm_callable)

    pad_mask = obs["timestep_pad_mask"]
    actions, dof_ids, chunk_steps, view_ids, mask_act = extract_bundled_actions(batch, cfg.horizon)

    def step(state: Any) -> tuple[Any, Any]:
        return train_step(
            state, obs, task, pad_mask, actions, dof_ids, chunk_steps, view_ids=view_ids, mask_act=mask_act
        )

    # Step 0: compile. Reuse the same batch afterwards so we time the device
    # step, not the data loader.
    t0 = time.perf_counter()
    state, info = step(state)
    jax.block_until_ready(info["loss"])
    result["compile_s"] = round(time.perf_counter() - t0, 1)

    times = []
    for _ in range(max(cfg.steps - 1, 3)):
        t0 = time.perf_counter()
        state, info = step(state)
        jax.block_until_ready(info["loss"])
        times.append(time.perf_counter() - t0)

    step_s = sorted(times)[len(times) // 2]  # median
    result["step_ms"] = round(step_s * 1000, 1)
    result["samples_per_s"] = round(batch_size / step_s, 1)
    result["peak_gib"] = round(_peak_memory_gib(), 2)
    result["loss"] = float(info["loss"])
    return result


# -- parent: sweep ------------------------------------------------------------


_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "OOM", "XlaRuntimeError: Resource exhausted")


def _launch(cfg: Config, size: str, batch: int, image: str) -> dict[str, Any]:
    result_path = Path(cfg.out).with_suffix(f".child.{size}.{batch}.{image}.json")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mix", cfg.mix,
        "--sizes", size,
        "--batch-sizes", str(batch),
        "--image-sizes", image,
        "--horizon", str(cfg.horizon),
        "--steps", str(cfg.steps),
        "--mp", str(cfg.mp),
        "--child-result", str(result_path),
    ]
    base = {"size": size, "batch_size": batch, "image": image}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout)
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout"}
    if result_path.exists():
        row = json.loads(result_path.read_text())
        result_path.unlink()
        return row
    tail = (proc.stderr or "")[-4000:]
    status = "oom" if any(m in tail for m in _OOM_MARKERS) else "error"
    return {**base, "status": status, "error": tail.splitlines()[-1] if tail else f"exit={proc.returncode}"}


def main(cfg: Config) -> None:
    if cfg.child_result is not None:
        row: dict[str, Any]
        try:
            row = run_one(cfg)
        except Exception as e:
            status = "oom" if any(m in repr(e) for m in _OOM_MARKERS) else "error"
            row = {
                "size": cfg.sizes[0],
                "batch_size": cfg.batch_sizes[0],
                "image": cfg.image_sizes[0],
                "status": status,
                "error": repr(e)[:500],
            }
        Path(cfg.child_result).write_text(json.dumps(row))
        print(json.dumps(row, indent=2))
        return

    from rich import print as rprint
    from rich.table import Table

    combos = [(s, b, i) for s in cfg.sizes for i in cfg.image_sizes for b in sorted(cfg.batch_sizes)]
    rprint(f"[bold]model_sizer[/]: mix={cfg.mix} horizon={cfg.horizon} — {len(combos)} combos")
    if cfg.dry:
        for c in combos:
            rprint(f"  size={c[0]} image={c[2]} batch={c[1]}")
        return

    rows: list[dict[str, Any]] = []
    dead: set[tuple[str, str]] = set()  # (size, image) that OOM'd — skip bigger batches
    try:
        for n, (size, batch, image) in enumerate(combos, 1):
            if (size, image) in dead:
                rows.append({"size": size, "batch_size": batch, "image": image, "status": "skipped (smaller batch OOM'd)"})
                continue
            rprint(f"[dim]\\[{n}/{len(combos)}][/] size={size} image={image} batch={batch} ...")
            row = _launch(cfg, size, batch, image)
            rows.append(row)
            if row["status"] in ("oom", "timeout"):
                dead.add((size, image))
            rprint(f"    -> {row['status']}"
                   + (f"  step={row.get('step_ms')}ms  {row.get('samples_per_s')} samples/s"
                      f"  peak={row.get('peak_gib')}GiB" if row["status"] == "ok" else ""))
            Path(cfg.out).write_text(json.dumps(rows, indent=2))
    except KeyboardInterrupt:
        rprint("[yellow]interrupted — reporting partial results[/]")

    Path(cfg.out).write_text(json.dumps(rows, indent=2))
    table = Table(title=f"model sizer — {cfg.mix} (horizon={cfg.horizon})")
    for col in ("size", "image", "batch", "params", "trainable", "step ms", "samples/s", "peak GiB", "compile s", "status"):
        table.add_column(col, justify="right")
    for r in rows:
        fmt = lambda k, d=1e6, suf="M": f"{r[k] / d:.1f}{suf}" if k in r else "-"
        table.add_row(
            r["size"],
            r["image"],
            str(r["batch_size"]),
            fmt("n_params"),
            fmt("n_trainable"),
            str(r.get("step_ms", "-")),
            str(r.get("samples_per_s", "-")),
            str(r.get("peak_gib", "-")),
            str(r.get("compile_s", "-")),
            r["status"],
        )
    rprint(table)
    rprint(f"results -> {cfg.out}")


if __name__ == "__main__":
    main(tyro.cli(Config))
