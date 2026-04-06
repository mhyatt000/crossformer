"""Benchmark ArrayRecord read for the real access pattern:

  sample(T) = images[T] + proprio[T:T+50]

Strategy C with independent group_size tuning per modality:
  - images: random access (1 read) → small group_size likely best
  - proprio: contiguous window (50 reads) → larger group_size may help
"""
from __future__ import annotations

import shutil
import time
from itertools import product
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
ROOT = Path("/tmp/arec_bench3")

IMG_OPTIONS = ["group_size:1", "group_size:8", "group_size:32"]
PRO_OPTIONS = ["group_size:1", "group_size:8", "group_size:32", "group_size:64", "group_size:128"]


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
    rng = np.random.default_rng(SEED)
    return [make_step(rng, ep, s) for ep in range(N_EPISODES) for s in range(STEPS_PER_EP)]


# ── writers ───────────────────────────────────────────────────────────────

def _write_shard(path: Path, records, opt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = ArrayRecordWriter(str(path), options=opt)
    for r in records:
        w.write(pack(r))
    w.close()


def build_split(steps, img_opt, pro_opt):
    """Build separate image and proprio shards with independent group_size."""
    tag = f"img_{img_opt.replace(':', '')}_pro_{pro_opt.replace(':', '')}"
    dest = ROOT / tag
    if dest.exists():
        shutil.rmtree(dest)

    img_path = dest / "images-00000.arrayrecord"
    pro_path = dest / "proprio-00000.arrayrecord"

    img_records = [{"image": s["image"], "info": s["info"]} for s in steps]
    pro_records = [{"proprio": s["proprio"], "info": s["info"]} for s in steps]

    _write_shard(img_path, img_records, img_opt)
    _write_shard(pro_path, pro_records, pro_opt)

    return (
        ArrayRecordDataSource([str(img_path)]),
        ArrayRecordDataSource([str(pro_path)]),
    )


# ── benchmark ─────────────────────────────────────────────────────────────

def sample_indices(n_steps, window, n_samples=200):
    rng = np.random.default_rng(0)
    return rng.integers(0, n_steps - window, size=n_samples)


def bench_split(img_src, pro_src, indices, rounds=5):
    """img[T] (1 random read) + proprio[T:T+50] (batched contiguous)."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            # 1 image read
            _ = unpack(img_src[t])
            # 50 proprio reads batched
            batch = pro_src.__getitems__(list(range(t, t + PROPRIO_WINDOW)))
            for b in batch:
                _ = unpack(b)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_img_only(img_src, indices, rounds=5):
    """Just the image read portion."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in indices:
            _ = unpack(img_src[int(t)])
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_pro_only(pro_src, indices, rounds=5):
    """Just the proprio window portion."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            batch = pro_src.__getitems__(list(range(t, t + PROPRIO_WINDOW)))
            for b in batch:
                _ = unpack(b)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    console.rule("Generating data")
    steps = gen_all()
    indices = sample_indices(N_STEPS, PROPRIO_WINDOW, n_samples=200)

    img_bytes = len(pack({"image": steps[0]["image"], "info": steps[0]["info"]}))
    pro_bytes = len(pack({"proprio": steps[0]["proprio"], "info": steps[0]["info"]}))
    console.print(f"img record: {img_bytes:,} B  |  proprio record: {pro_bytes:,} B")
    console.print(f"per sample: 1 img read + {PROPRIO_WINDOW} proprio reads (batched)")
    console.print(f"samples: {len(indices)}, rounds: 5\n")

    # ── Part 1: isolate each modality ─────────────────────────────────────

    console.rule("Part 1: Image read (1 random access per sample)")
    img_table = Table(title="Image-only timing")
    img_table.add_column("img group_size", style="cyan")
    img_table.add_column("time (s)", justify="right")
    img_table.add_column("samp/s", justify="right")

    img_results = {}
    for img_opt in IMG_OPTIONS:
        # build with dummy proprio option
        img_src, _ = build_split(steps, img_opt, "group_size:1")
        t = bench_img_only(img_src, indices)
        img_results[img_opt] = t
        img_table.add_row(img_opt, f"{t:.4f}", f"{len(indices) / t:.0f}")

    console.print(img_table)

    console.rule("Part 2: Proprio window (50 contiguous batched reads per sample)")
    pro_table = Table(title="Proprio-only timing")
    pro_table.add_column("pro group_size", style="cyan")
    pro_table.add_column("time (s)", justify="right")
    pro_table.add_column("samp/s", justify="right")

    pro_results = {}
    for pro_opt in PRO_OPTIONS:
        _, pro_src = build_split(steps, "group_size:1", pro_opt)
        t = bench_pro_only(pro_src, indices)
        pro_results[pro_opt] = t
        pro_table.add_row(pro_opt, f"{t:.4f}", f"{len(indices) / t:.0f}")

    console.print(pro_table)

    # ── Part 2: combined sweep ────────────────────────────────────────────

    console.rule("Part 3: Combined sweep (img + proprio)")
    combo_table = Table(title=f"img[T] + proprio[T:T+{PROPRIO_WINDOW}]  ({len(indices)} samples)")
    combo_table.add_column("img gs", style="cyan")
    combo_table.add_column("pro gs", style="cyan")
    combo_table.add_column("total (s)", justify="right")
    combo_table.add_column("samp/s", justify="right")
    combo_table.add_column("img frac", justify="right")
    combo_table.add_column("pro frac", justify="right")

    best_t, best_combo = float("inf"), ("", "")
    for img_opt, pro_opt in product(IMG_OPTIONS, PRO_OPTIONS):
        img_src, pro_src = build_split(steps, img_opt, pro_opt)
        t = bench_split(img_src, pro_src, indices)

        t_img = img_results[img_opt]
        t_pro = pro_results[pro_opt]
        img_frac = t_img / t if t > 0 else 0
        pro_frac = t_pro / t if t > 0 else 0

        if t < best_t:
            best_t, best_combo = t, (img_opt, pro_opt)

        combo_table.add_row(
            img_opt, pro_opt,
            f"{t:.4f}", f"{len(indices) / t:.0f}",
            f"{img_frac:.0%}", f"{pro_frac:.0%}",
        )

    console.print(combo_table)
    console.print(f"\n[bold green]Best:[/] img={best_combo[0]}  pro={best_combo[1]}  → {len(indices)/best_t:.0f} samp/s")


if __name__ == "__main__":
    main()
