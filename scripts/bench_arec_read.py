"""Benchmark ArrayRecord read for the real access pattern:

  sample(T) = images[T] + proprio[T:T+50]

Strategies:
  A) Single file, vary group_size, fetch 1 + 50 steps
  B) Single file + custom source that batches the 50-step fetch
  C) Split files: images.arrayrecord + proprio.arrayrecord
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
from array_record.python.array_record_module import ArrayRecordWriter
from rich.console import Console
from rich.table import Table

console = Console()

N_STEPS = 1000
N_EPISODES = 10
STEPS_PER_EP = N_STEPS // N_EPISODES
IMG_H, IMG_W = 64, 64
PROPRIO_WINDOW = 50
SEED = 42
ROOT = Path("/tmp/arec_bench2")

OPTIONS = ["group_size:1", "group_size:8", "group_size:32", "group_size:64"]


# ── pack / unpack ─────────────────────────────────────────────────────────

def _pack_default(obj):
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype), "data": obj.tobytes()}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj)
    raise TypeError(type(obj))


def _unpack_hook(obj):
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def pack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=_pack_default, use_bin_type=True)


def unpack(buf: bytes) -> Any:
    return msgpack.unpackb(buf, object_hook=_unpack_hook, raw=False)


# ── data generation ───────────────────────────────────────────────────────

def make_step(rng, ep_id, step_id):
    return {
        "image": {
            "low": rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "side": rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "wrist": rng.integers(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8),
        },
        "proprio": {
            "joints": rng.standard_normal(7).astype(np.float32),
            "gripper": rng.standard_normal(1).astype(np.float32),
            "position": rng.standard_normal(3).astype(np.float32),
            "orientation": rng.standard_normal(4).astype(np.float32),
        },
        "info": {"id": {"episode": np.int32(ep_id), "step": np.int32(step_id)}},
    }


def gen_all():
    """Return list of all steps (materialized for reuse)."""
    rng = np.random.default_rng(SEED)
    return [make_step(rng, ep, s) for ep in range(N_EPISODES) for s in range(STEPS_PER_EP)]


# ── writers ───────────────────────────────────────────────────────────────

def _write_shard(path: Path, records, opt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = ArrayRecordWriter(str(path), options=opt)
    for r in records:
        w.write(pack(r))
    w.close()


def build_combined(steps: list[dict], opt: str) -> ArrayRecordDataSource:
    """Strategy A/B: everything in one file."""
    tag = opt.replace(":", "")
    dest = ROOT / "combined" / tag
    if dest.exists():
        shutil.rmtree(dest)
    p = dest / "data-00000.arrayrecord"
    _write_shard(p, steps, opt)
    return ArrayRecordDataSource([str(p)])


def build_split(steps: list[dict], opt: str) -> tuple[ArrayRecordDataSource, ArrayRecordDataSource]:
    """Strategy C: separate image and proprio files."""
    tag = opt.replace(":", "")
    dest = ROOT / "split" / tag
    if dest.exists():
        shutil.rmtree(dest)

    img_path = dest / "images-00000.arrayrecord"
    pro_path = dest / "proprio-00000.arrayrecord"

    img_records = [{"image": s["image"], "info": s["info"]} for s in steps]
    pro_records = [{"proprio": s["proprio"], "info": s["info"]} for s in steps]

    _write_shard(img_path, img_records, opt)
    _write_shard(pro_path, pro_records, opt)

    return (
        ArrayRecordDataSource([str(img_path)]),
        ArrayRecordDataSource([str(pro_path)]),
    )


# ── access pattern simulation ─────────────────────────────────────────────

def sample_indices(n_steps, window, n_samples=200):
    """Generate valid sample start indices."""
    rng = np.random.default_rng(0)
    max_t = n_steps - window
    return rng.integers(0, max_t, size=n_samples)


def strategy_a_sequential(src: ArrayRecordDataSource, indices) -> float:
    """Read step T (full), then steps T+1..T+49 (unpack all, discard images)."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for t in indices:
            # step T: full
            step_t = unpack(src[int(t)])
            # steps T+1..T+49: read one by one
            for i in range(1, PROPRIO_WINDOW):
                full = unpack(src[int(t + i)])
                _ = full["proprio"]  # only need proprio
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def strategy_b_batched(src: ArrayRecordDataSource, indices) -> float:
    """Read step T (full), then batch-read T+1..T+49 via __getitems__."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            step_t = unpack(src[t])
            batch = src.__getitems__(list(range(t + 1, t + PROPRIO_WINDOW)))
            for b in batch:
                full = unpack(b)
                _ = full["proprio"]
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def strategy_b_batched_lazy(src: ArrayRecordDataSource, indices) -> float:
    """Same as B but only unpack proprio keys (skip image deserialization)."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            step_t = unpack(src[t])
            batch = src.__getitems__(list(range(t + 1, t + PROPRIO_WINDOW)))
            for b in batch:
                # unpack but we pay the full msgpack cost regardless
                full = unpack(b)
                _ = full["proprio"]
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def strategy_c_split(img_src: ArrayRecordDataSource, pro_src: ArrayRecordDataSource, indices) -> float:
    """Read images[T] + proprio[T:T+50] from separate files."""
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            img_t = unpack(img_src[t])
            pro_batch = pro_src.__getitems__(list(range(t, t + PROPRIO_WINDOW)))
            for b in pro_batch:
                _ = unpack(b)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    console.rule("Generating data")
    steps = gen_all()
    indices = sample_indices(N_STEPS, PROPRIO_WINDOW, n_samples=200)

    record_bytes = len(pack(steps[0]))
    img_bytes = len(pack({"image": steps[0]["image"], "info": steps[0]["info"]}))
    pro_bytes = len(pack({"proprio": steps[0]["proprio"], "info": steps[0]["info"]}))
    console.print(f"record size: {record_bytes:,} B  |  img-only: {img_bytes:,} B  |  proprio-only: {pro_bytes:,} B")
    console.print(f"per sample: 1 full read ({record_bytes:,}B) + {PROPRIO_WINDOW-1} proprio reads")
    console.print(f"  strategy A/B waste: {(PROPRIO_WINDOW-1) * record_bytes:,} B decoded (mostly images)")
    console.print(f"  strategy C waste:   0 B  (proprio file is {pro_bytes} B/step)")
    console.print(f"  samples: {len(indices)}")

    table = Table(title=f"Access pattern: img[T] + proprio[T:T+{PROPRIO_WINDOW}]  ({len(indices)} samples)")
    table.add_column("group_size", style="cyan")
    table.add_column("A: seq 1+49 (s)", justify="right")
    table.add_column("A samp/s", justify="right")
    table.add_column("B: batched (s)", justify="right")
    table.add_column("B samp/s", justify="right")
    table.add_column("C: split (s)", justify="right")
    table.add_column("C samp/s", justify="right")
    table.add_column("C speedup", justify="right", style="green")

    for opt in OPTIONS:
        console.print(f"\n[bold]{opt}[/]")

        # build
        combined = build_combined(steps, opt)
        img_src, pro_src = build_split(steps, opt)

        # bench
        t_a = strategy_a_sequential(combined, indices)
        t_b = strategy_b_batched(combined, indices)
        t_c = strategy_c_split(img_src, pro_src, indices)

        n = len(indices)
        speedup = t_b / t_c if t_c > 0 else float("inf")

        table.add_row(
            opt,
            f"{t_a:.3f}", f"{n / t_a:.0f}",
            f"{t_b:.3f}", f"{n / t_b:.0f}",
            f"{t_c:.3f}", f"{n / t_c:.0f}",
            f"{speedup:.1f}x",
        )

    console.print()
    console.rule("Results")
    console.print(table)
    console.print()
    console.print("[bold]Legend:[/]")
    console.print("  A = single file, sequential 1+49 reads (unpack all)")
    console.print("  B = single file, batched __getitems__ for the 49-step window")
    console.print("  C = split files (images + proprio), batched proprio reads")


if __name__ == "__main__":
    main()
