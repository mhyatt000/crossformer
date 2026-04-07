"""Benchmark ArrayRecord read for the real access pattern:

  sample(T) = images[T] + proprio[T:T+50]

Strategy C: split files with independent group_size per modality.
Sweeps group_size combos, proprio batch chunk sizes, and threaded loading.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
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

IMG_OPTIONS = ["group_size:1", "group_size:8"]
PRO_OPTIONS = ["group_size:1", "group_size:8", "group_size:32", "group_size:64"]
CHUNK_SIZES = [10, 25, 50]  # how many proprio steps to batch per __getitems__ call


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


def _sample_goal_offset(rng, max_offset, decay=0.05):
    if max_offset <= 0:
        return 0
    return min(int(rng.geometric(decay)), max_offset)


class _MultiArrayRecordSource:
    """Mirror of MultiArrayRecordSource for testing without heavy imports."""

    def __init__(self, img_src, pro_src, window=50, goal=False, goal_decay=0.05, seed=0):
        assert len(img_src) == len(pro_src)
        self._img, self._pro = img_src, pro_src
        self._window = window
        self._goal = goal
        self._goal_decay = goal_decay
        self._rng = np.random.default_rng(seed)
        self._n = len(img_src)

    def __len__(self):
        return self._n - self._window + 1

    def __getitem__(self, i):
        img_rec = unpack(self._img[i])
        end = min(i + self._window, self._n)
        pro_recs = [unpack(b) for b in self._pro.__getitems__(list(range(i, end)))]
        pro_stacked = _tree_stack([r["proprio"] for r in pro_recs])
        out = {**img_rec, "proprio": pro_stacked}
        if self._goal:
            max_offset = self._n - 1 - i
            offset = _sample_goal_offset(self._rng, max_offset, self._goal_decay)
            goal_rec = unpack(self._img[i + offset])
            out["goal"] = goal_rec["image"]
        return out

    def __getitems__(self, indices):
        return [self[i] for i in indices]


def _tree_stack(dicts):
    keys = dicts[0].keys()
    return {k: np.stack([d[k] for d in dicts]) for k in keys}


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
    tag = f"img_{img_opt.replace(':', '')}_pro_{pro_opt.replace(':', '')}"
    dest = ROOT / tag
    if dest.exists():
        shutil.rmtree(dest)

    img_path = dest / "images-00000.arrayrecord"
    pro_path = dest / "proprio-00000.arrayrecord"

    _write_shard(img_path, [{"image": s["image"], "info": s["info"]} for s in steps], img_opt)
    _write_shard(pro_path, [{"proprio": s["proprio"], "info": s["info"]} for s in steps], pro_opt)

    return (
        ArrayRecordDataSource([str(img_path)]),
        ArrayRecordDataSource([str(pro_path)]),
    )


# ── benchmark helpers ─────────────────────────────────────────────────────

def sample_indices(n_steps, window, n_samples=200):
    rng = np.random.default_rng(0)
    return rng.integers(0, n_steps - window, size=n_samples)


def bench_sequential(img_src, pro_src, indices, chunk=50, rounds=5):
    """Sequential: read img then proprio."""
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            _ = unpack(img_src[t])
            idxs = list(range(t, t + PROPRIO_WINDOW))
            for s in range(0, len(idxs), chunk):
                batch = pro_src.__getitems__(idxs[s:s + chunk])
                for b in batch:
                    _ = unpack(b)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_threaded(img_src, pro_src, indices, chunk=50, rounds=5):
    """Threaded: img and proprio reads overlap via ThreadPoolExecutor."""
    pool = ThreadPoolExecutor(max_workers=2)

    def read_img(t):
        return unpack(img_src[t])

    def read_pro(t):
        idxs = list(range(t, t + PROPRIO_WINDOW))
        out = []
        for s in range(0, len(idxs), chunk):
            batch = pro_src.__getitems__(idxs[s:s + chunk])
            out.extend(unpack(b) for b in batch)
        return out

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for t in indices:
            t = int(t)
            f_img = pool.submit(read_img, t)
            f_pro = pool.submit(read_pro, t)
            _ = f_img.result()
            _ = f_pro.result()
        times.append(time.perf_counter() - t0)
    pool.shutdown(wait=False)
    return sum(times) / len(times)


def bench_threaded_prefetch(img_src, pro_src, indices, chunk=50, rounds=5, prefetch=4):
    """Submit prefetch samples ahead, drain as we go."""
    pool = ThreadPoolExecutor(max_workers=prefetch)

    def load_one(t):
        t = int(t)
        img = unpack(img_src[t])
        idxs = list(range(t, t + PROPRIO_WINDOW))
        pros = []
        for s in range(0, len(idxs), chunk):
            batch = pro_src.__getitems__(idxs[s:s + chunk])
            pros.extend(unpack(b) for b in batch)
        return img, pros

    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        futures = []
        for i, t in enumerate(indices):
            futures.append(pool.submit(load_one, t))
            # drain completed to bound memory
            if len(futures) >= prefetch:
                _ = futures.pop(0).result()
        for f in futures:
            _ = f.result()
        times.append(time.perf_counter() - t0)
    pool.shutdown(wait=False)
    return sum(times) / len(times)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    console.rule("Generating data")
    steps = gen_all()
    indices = sample_indices(N_STEPS, PROPRIO_WINDOW, n_samples=200)
    n = len(indices)

    img_bytes = len(pack({"image": steps[0]["image"], "info": steps[0]["info"]}))
    pro_bytes = len(pack({"proprio": steps[0]["proprio"], "info": steps[0]["info"]}))
    console.print(f"img record: {img_bytes:,} B  |  proprio record: {pro_bytes:,} B")
    console.print(f"per sample: 1 img read + {PROPRIO_WINDOW} proprio reads")
    console.print(f"samples: {n}, rounds: 5\n")

    # ── sweep: group_size x chunk_size x load strategy ────────────────────

    console.rule("Full sweep")

    table = Table(title=f"img[T] + proprio[T:T+{PROPRIO_WINDOW}]  ({n} samples)")
    table.add_column("img gs", style="cyan")
    table.add_column("pro gs", style="cyan")
    table.add_column("chunk", justify="right")
    table.add_column("seq (s)", justify="right")
    table.add_column("seq s/s", justify="right")
    table.add_column("thr (s)", justify="right")
    table.add_column("thr s/s", justify="right")
    table.add_column("pre4 (s)", justify="right")
    table.add_column("pre4 s/s", justify="right", style="green")

    best_t, best_cfg = float("inf"), {}

    for img_opt, pro_opt in product(IMG_OPTIONS, PRO_OPTIONS):
        console.print(f"  building img={img_opt} pro={pro_opt}")
        img_src, pro_src = build_split(steps, img_opt, pro_opt)

        for chunk in CHUNK_SIZES:
            t_seq = bench_sequential(img_src, pro_src, indices, chunk=chunk)
            t_thr = bench_threaded(img_src, pro_src, indices, chunk=chunk)
            t_pre = bench_threaded_prefetch(img_src, pro_src, indices, chunk=chunk)

            best_here = min(t_seq, t_thr, t_pre)
            if best_here < best_t:
                best_t = best_here
                best_cfg = {"img": img_opt, "pro": pro_opt, "chunk": chunk,
                            "method": ["seq", "thr", "pre4"][[t_seq, t_thr, t_pre].index(best_here)]}

            table.add_row(
                img_opt, pro_opt, str(chunk),
                f"{t_seq:.3f}", f"{n / t_seq:.0f}",
                f"{t_thr:.3f}", f"{n / t_thr:.0f}",
                f"{t_pre:.3f}", f"{n / t_pre:.0f}",
            )

    console.print()
    console.rule("Results")
    console.print(table)
    console.print(
        f"\n[bold green]Best:[/] img={best_cfg['img']}  pro={best_cfg['pro']}"
        f"  chunk={best_cfg['chunk']}  method={best_cfg['method']}"
        f"  → {n / best_t:.0f} samp/s"
    )

    # ── MultiArrayRecordSource test ─────────────────────────────────────
    console.print()
    console.rule("MultiArrayRecordSource")

    MultiArrayRecordSource = _MultiArrayRecordSource

    img_src, pro_src = build_split(steps, best_cfg["img"], best_cfg["pro"])

    for use_goal in [False, True]:
        label = "goal=True" if use_goal else "goal=False"
        multi = MultiArrayRecordSource(img_src, pro_src, window=PROPRIO_WINDOW, goal=use_goal)
        console.print(f"\n[bold]{label}[/]  len={len(multi)}")

        valid = indices[indices < len(multi)]
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for i in valid:
                _ = multi[int(i)]
            times.append(time.perf_counter() - t0)
        avg = sum(times) / len(times)
        console.print(f"  {len(valid)/avg:.0f} samp/s  ({avg:.3f}s for {len(valid)} samples)")

        sample = multi[int(valid[0])]
        console.print(f"  [bold]output spec:[/]")
        _show_spec(sample)


def _show_spec(d, prefix="", label=None):
    if label:
        console.print(f"  [{label}]")
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _show_spec(v, prefix=f"{key}.")
        elif isinstance(v, np.ndarray):
            console.print(f"    {key:35s} {str(v.dtype):10s} {v.shape}")
        else:
            console.print(f"    {key:35s} {type(v).__name__:10s} {v}")


if __name__ == "__main__":
    main()
