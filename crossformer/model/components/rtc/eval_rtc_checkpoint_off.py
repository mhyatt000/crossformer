"""Offline eval entry point: load checkpoint + dataset,

This script is NOT a test file. It is the entry point that:
  1. Loads the checkpoint and dataset.
  2. Runs eval functions and prints results.
  3. Saves a figure.

The actual pytest test logic lives in:
  test_rtc_integration.py      → (3) inference quality / region ordering
  test_rtc_chunk_continuity.py → (1) boundary jump, (2) jitter, (4) figure

"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crossformer.model.crossformer_model import CrossFormerModel
from crossformer.model.components.rtc.rtc_algorithm import guided_inference

# ---------------------------------------------------------------------------
# RTC parameters (paper Table 4 / checkpoint config)
# ---------------------------------------------------------------------------
H          = 20
MAX_A      = 14
FLOW_STEPS = 50
BETA       = 5.0
D          = 1
S          = 3
SEEDS      = [0, 1, 2, 3, 4]
DELAY_SWEEP_D = [1, 2, 3]

DOF_JOINTS      = slice(0, 7)
DOF_GRIPPER     = slice(7, 8)
DOF_POSITION    = slice(8, 11)
DOF_ORIENTATION = slice(11, 14)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _iter_arrayrecord(dataset_path: str, max_records: int):
    from array_record.python.array_record_data_source import ArrayRecordDataSource
    from crossformer.data.arec.arec import unpack_record

    path = Path(dataset_path).expanduser()
    shards = sorted(path.glob("data-*.arrayrecord"))
    assert shards, f"No arrayrecord shards found in {path}"

    source = ArrayRecordDataSource([str(s) for s in shards])
    for i in range(min(max_records, len(source))):
        yield unpack_record(source[i])

def _record_to_action(record: dict) -> np.ndarray:
    """Pack dataset action chunk into (H, MAX_A) array."""
    act = record["action"]
    a = np.zeros((H, MAX_A), dtype=np.float32)
    a[:, DOF_JOINTS]      = act["joints"]       # (20, 7)
    a[:, DOF_GRIPPER]     = act["gripper"]      # (20, 1)
    a[:, DOF_POSITION]    = act["position"]     # (20, 3)
    a[:, DOF_ORIENTATION] = act["orientation"]  # (20, 3)
    return a  # (20, 14)

def _record_to_raw_obs(record: dict) -> tuple[dict, dict]:
    """Convert one ArrayRecord step into (obs_raw, task_raw) numpy dicts."""
    obs  = record["observation"]
    prop = obs["proprio"]
    info = record["info"]

    def resize(img224: np.ndarray) -> np.ndarray:
        return cv2.resize(img224, (64, 64), interpolation=cv2.INTER_AREA)

    def img(key: str) -> np.ndarray:
        return resize(obs["image"][key])[None, None]

    def p(key: str) -> np.ndarray:
        return prop[key].astype(np.float32)[None, None]

    true_mask = np.ones((1, 1), dtype=bool)
    pad_mask_dict = {k: true_mask for k in [
        "image_primary", "image_side", "image_left_wrist",
        "proprio_joints", "proprio_gripper", "proprio_position",
        "proprio_orientation", "time", "timestep",
    ]}

    obs_raw = {
        "image_primary":       img("low"),
        "image_side":          img("side"),
        "image_left_wrist":    img("wrist"),
        "proprio_joints":      p("joints"),
        "proprio_gripper":     p("gripper"),
        "proprio_position":    p("position"),
        "proprio_orientation": p("orientation"),
        "time":                obs["time"].astype(np.float32)[None],
        "timestep":            np.array([[int(info["step"])]], dtype=np.int32),
        "timestep_pad_mask":   true_mask,
        "pad_mask_dict":       pad_mask_dict,
    }

    task_raw = {
        "language_instruction": record["language_embedding"].astype(np.float32)[None],
        "pad_mask_dict": {"language_instruction": np.ones((1, 1), dtype=bool)},
    }

    return obs_raw, task_raw


def load_obs_list(model: CrossFormerModel, dataset_path: str, n_steps: int) -> list:
    """Load n_steps. Returns list of (obs_raw, task_raw, action_np)."""
    print(f"  Loading {n_steps} steps from {dataset_path} ...")
    records = []
    for rec in _iter_arrayrecord(dataset_path, n_steps):
        obs_raw, task_raw = _record_to_raw_obs(rec)
        action = _record_to_action(rec)
        records.append((obs_raw, task_raw, action))
    print(f"  Loaded {len(records)} steps.\n")
    return records


# ---------------------------------------------------------------------------
# Obs building
# ---------------------------------------------------------------------------

def _make_obs(model: CrossFormerModel, obs_raw: dict, task_raw: dict,
              rng: jax.Array) -> dict:
    """Run transformer on one obs, return dict ready for guided_inference."""
    obs_jax  = jax.tree_util.tree_map(jnp.array, obs_raw)
    task_jax = jax.tree_util.tree_map(jnp.array, task_raw)
    pad_mask = obs_jax["timestep_pad_mask"]
    transformer_outputs = model.run_transformer(obs_jax, task_jax, pad_mask, train=False)
    return {
        "transformer_outputs": transformer_outputs,
        "dof_ids":             jnp.zeros((1, MAX_A), dtype=jnp.int32),
        "chunk_steps":         jnp.zeros((1, H),     dtype=jnp.float32),
        "slot_pos":            None,
        "guide_input":         None,
        "guidance_mask":       None,
        "train":               False,
        "B": 1, "W": 1,
        "rng": rng,
    }


def _make_obs_synthetic(model: CrossFormerModel, rng: jax.Array) -> dict:
    obs      = model.example_batch["observation"]
    task     = model.example_batch["task"]
    pad_mask = obs["timestep_pad_mask"]
    transformer_outputs = model.run_transformer(obs, task, pad_mask, train=False)
    return {
        "transformer_outputs": transformer_outputs,
        "dof_ids":             jnp.zeros((1, MAX_A), dtype=jnp.int32),
        "chunk_steps":         jnp.zeros((1, H),     dtype=jnp.float32),
        "slot_pos":            None,
        "guide_input":         None,
        "guidance_mask":       None,
        "train":               False,
        "B": 1, "W": 1,
        "rng": rng,
    }


def _get_obs_and_aprev(model, obs_list, i):
    obs_raw, task_raw, action = obs_list[i]
    obs = _make_obs(model, obs_raw, task_raw, jax.random.PRNGKey(0))
    return obs, action  # (H, MAX_A)


# ---------------------------------------------------------------------------
# Inference helpers (also imported by test files via conftest)
# ---------------------------------------------------------------------------

def run_guided(pi, obs: dict, A_prev: jax.Array, seed: int,
               d: int = D, s: int = S) -> np.ndarray:
    out = guided_inference(
        pi, {**obs, "rng": jax.random.PRNGKey(seed)},
        A_prev, d=d, s=s, flow_steps=FLOW_STEPS, beta=BETA,
    )
    return np.array(out[0, 0])


def run_naive(pi, obs: dict, seed: int) -> np.ndarray:
    out = pi.predict_action(
        obs["transformer_outputs"],
        rng=jax.random.PRNGKey(seed),
        dof_ids=obs["dof_ids"],
        chunk_steps=obs["chunk_steps"],
        train=False,
    )
    return np.array(out[0, 0])


def region_l2(A_new: np.ndarray, A_ref: np.ndarray, lo: int, hi: int) -> float:
    return float(np.mean(np.linalg.norm(A_new[lo:hi] - A_ref[lo:hi], axis=-1)))


def boundary_jump(chunk_a: np.ndarray, chunk_b: np.ndarray, s: int = S) -> float:
    return float(np.linalg.norm(chunk_a[s - 1] - chunk_b[0]))


def print_stats(label: str, values: list):
    arr = np.array(values)
    print(f"  {label}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")


def make_step_iter(obs_list, model):
    if obs_list:
        return [_get_obs_and_aprev(model, obs_list, i) for i in range(len(obs_list))]
    obs = _make_obs_synthetic(model, jax.random.PRNGKey(42))
    return [(obs, np.ones((H, MAX_A), dtype=np.float32) * 0.5)]


# ---------------------------------------------------------------------------
# Eval functions
# ---------------------------------------------------------------------------

def eval_inference_quality(pi, obs_list, model) -> bool:
    """(3): frozen < intermediate < fresh error ordering."""
    src = f"dataset ({len(obs_list)} steps)" if obs_list else "synthetic"
    print(f"\n=== (3) Inference Quality — Region Error Ordering ===")
    print(f"    H={H}  d={D}  s={S}  flow_steps={FLOW_STEPS}  seeds={SEEDS}  obs={src}\n")

    passed = total = 0
    frozen_all, inter_all, fresh_all = [], [], []

    for step_idx, (obs, A_prev_np) in enumerate(make_step_iter(obs_list, model)):
        A_prev = jnp.array(A_prev_np)
        for seed in SEEDS:
            A_new = run_guided(pi, obs, A_prev, seed)
            fe  = region_l2(A_new, A_prev_np, 0,   D)
            ie  = region_l2(A_new, A_prev_np, D,   H - S)
            fre = region_l2(A_new, A_prev_np, H-S, H)
            frozen_all.append(fe); inter_all.append(ie); fresh_all.append(fre)
            ok = fe < ie < fre
            total += 1
            if ok: passed += 1
            print(f"  step={step_idx:03d}  seed={seed}  {'PASS' if ok else 'FAIL'}"
                  f"  frozen={fe:.4f}  inter={ie:.4f}  fresh={fre:.4f}")

    print(f"\n  Aggregated ({total} evals):")
    print_stats("frozen      ", frozen_all)
    print_stats("intermediate", inter_all)
    print_stats("fresh       ", fresh_all)
    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def eval_boundary_jump(pi, obs_list, model):
    """(1): RTC boundary jump < naive."""
    src = f"dataset ({len(obs_list)} steps)" if obs_list else "synthetic"
    print(f"\n=== (1) Plan Commitment — Boundary Jump RTC vs Naive ===")
    print(f"    H={H}  d={D}  s={S}  flow_steps={FLOW_STEPS}  seeds={SEEDS}  obs={src}\n")

    rtc_jumps, naive_jumps = [], []
    for step_idx, (obs, A_prev_np) in enumerate(make_step_iter(obs_list, model)):
        A_prev = jnp.array(A_prev_np)
        for seed in SEEDS:
            A_rtc   = run_guided(pi, obs, A_prev, seed)
            A_naive = run_naive(pi, obs, seed)
            rj = boundary_jump(A_prev_np, A_rtc)
            nj = boundary_jump(A_prev_np, A_naive)
            rtc_jumps.append(rj); naive_jumps.append(nj)
            print(f"  step={step_idx:03d}  seed={seed}  {'PASS' if rj < nj else 'FAIL'}"
                  f"  RTC={rj:.4f}  naive={nj:.4f}")

    print()
    print_stats("RTC   boundary jump", rtc_jumps)
    print_stats("naive boundary jump", naive_jumps)
    overall = np.mean(rtc_jumps) < np.mean(naive_jumps)
    print(f"\n  Result: {'PASS' if overall else 'FAIL'} — "
          f"RTC {'<' if overall else '>='} naive (mean)")
    return overall, rtc_jumps, naive_jumps


def eval_plan_commitment(pi, obs_list, model) -> bool:
    """(2): frozen region stability (jitter test)."""
    src = f"dataset ({len(obs_list)} steps)" if obs_list else "synthetic"
    print(f"\n=== (2) Jitter Test — Frozen Region Stability ===")
    print(f"    H={H}  d={D}  s={S}  flow_steps={FLOW_STEPS}  seeds={SEEDS}  obs={src}")
    print(f"    (a) frozen_rtc < frozen_naive   (b) frozen_rtc < fresh_rtc\n")

    pass_a = pass_b = total = 0
    frozen_rtc_all, frozen_naive_all, fresh_rtc_all = [], [], []

    for step_idx, (obs, A_prev_np) in enumerate(make_step_iter(obs_list, model)):
        A_prev = jnp.array(A_prev_np)
        for seed in SEEDS:
            A_rtc   = run_guided(pi, obs, A_prev, seed)
            A_naive = run_naive(pi, obs, seed)
            frozen_rtc   = float(np.linalg.norm(A_rtc[:D]   - A_prev_np[:D]))
            frozen_naive = float(np.linalg.norm(A_naive[:D] - A_prev_np[:D]))
            fresh_rtc    = float(np.mean(np.linalg.norm(
                A_rtc[H-S:H] - A_prev_np[H-S:H], axis=-1)))
            frozen_rtc_all.append(frozen_rtc)
            frozen_naive_all.append(frozen_naive)
            fresh_rtc_all.append(fresh_rtc)
            ok_a = frozen_rtc < frozen_naive
            ok_b = frozen_rtc < fresh_rtc
            total += 1
            if ok_a: pass_a += 1
            if ok_b: pass_b += 1
            print(f"  step={step_idx:03d}  seed={seed}  "
                  f"(a){'PASS' if ok_a else 'FAIL'} (b){'PASS' if ok_b else 'FAIL'}"
                  f"  frozen_rtc={frozen_rtc:.4f}"
                  f"  frozen_naive={frozen_naive:.4f}"
                  f"  fresh_rtc={fresh_rtc:.4f}")

    print(f"\n  Aggregated ({total} evals):")
    print_stats("frozen_rtc  ", frozen_rtc_all)
    print_stats("frozen_naive", frozen_naive_all)
    print_stats("fresh_rtc   ", fresh_rtc_all)
    print(f"\n  (a) frozen_rtc < frozen_naive : {pass_a}/{total}")
    print(f"  (b) frozen_rtc < fresh_rtc    : {pass_b}/{total}")
    overall = (pass_a == total) and (pass_b == total)
    print(f"\n  Result: {'PASS' if overall else 'FAIL'}")
    return overall


def eval_delay_sweep(pi, obs_list, model) -> list:
    """Delay sweep d in {1,2,3}."""
    print(f"\n=== Delay Sweep: d in {{1,2,3}}, s in [d, H-d] ===")
    print("    Metric: frozen_err vs fresh_err (ratio = fresh/frozen)\n")

    if obs_list:
        obs, A_prev_np = _get_obs_and_aprev(model, obs_list, 0)
        print("    (using first dataset step)\n")
    else:
        obs = _make_obs_synthetic(model, jax.random.PRNGKey(42))
        A_prev_np = np.ones((H, MAX_A), dtype=np.float32) * 0.5

    A_prev  = jnp.array(A_prev_np)
    results = []

    for d in DELAY_SWEEP_D:
        for s in range(d, min(H - d + 1, d + 5)):
            frozen_errs, fresh_errs, order_pass = [], [], 0
            for seed in SEEDS:
                A_new = run_guided(pi, obs, A_prev, seed, d=d, s=s)
                fe  = region_l2(A_new, A_prev_np, 0,   d)
                fre = region_l2(A_new, A_prev_np, H-s, H)
                frozen_errs.append(fe); fresh_errs.append(fre)
                if fe < fre: order_pass += 1
            mf  = float(np.mean(frozen_errs)); sf = float(np.std(frozen_errs))
            mfr = float(np.mean(fresh_errs));  sfr = float(np.std(fresh_errs))
            ratio    = mfr / (mf + 1e-8)
            order_ok = order_pass == len(SEEDS)
            print(f"  d={d}  s={s}  {'PASS' if order_ok else 'FAIL'}  "
                  f"frozen={mf:.4f}+-{sf:.4f}  fresh={mfr:.4f}+-{sfr:.4f}  "
                  f"ratio={ratio:.1f}x  ordering={order_pass}/{len(SEEDS)}")
            results.append(dict(d=d, s=s, mean_frozen=mf, std_frozen=sf,
                                mean_fresh=mfr, std_fresh=sfr,
                                ratio=ratio, order_ok=order_ok))

    n_pass = sum(1 for r in results if r["order_ok"])
    print(f"\n  Result: {n_pass}/{len(results)} (d,s) pairs passed")
    return results


def eval_figure(pi, obs_list, model, out_path: Path):
    """(4): consecutive chunks figure."""
    print(f"\n=== (4) Figure — Consecutive Chunks ===")
    DOF_IDX = 0

    if obs_list:
        obs, A_prev_np = _get_obs_and_aprev(model, obs_list, 0)
        print("    (using first dataset step, same obs for all chunks)\n")
    else:
        obs = _make_obs_synthetic(model, jax.random.PRNGKey(42))
        A_prev_np = np.ones((H, MAX_A), np.float32) * 0.5

    A_prev = jnp.array(A_prev_np)
    rtc_chunks, naive_chunks = [], []
    for i in range(3):
        rtc_chunk = run_guided(pi, obs, A_prev, seed=i)
        rtc_chunks.append(rtc_chunk[:, DOF_IDX])
        A_prev = jnp.array(rtc_chunk)
        naive_chunks.append(run_naive(pi, obs, seed=i)[:, DOF_IDX])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    fig.suptitle("RTC vs Naive Async — Consecutive Chunks "
                 "(DOF 0 = joints[0], real checkpoint)", fontsize=12)
    colors = ["#2d6a4f", "#52b788", "#b7e4c7"]
    for ax, chunks, title in zip(
        axes, [rtc_chunks, naive_chunks], ["RTC (ours)", "Naive Async"]
    ):
        for i, (chunk, color) in enumerate(zip(chunks, colors)):
            x = np.arange(len(chunk))
            ax.plot(x, chunk, color=color, linewidth=2,
                    alpha=1.0 - i * 0.2, label=f"chunk {i+1}")
            ax.scatter(x, chunk, color=color, s=35, zorder=3,
                       alpha=1.0 - i * 0.2)
        ax.axvspan(-0.5,      D - 0.5,   color="#f4a261", alpha=0.15,
                   label=f"frozen (d={D})")
        ax.axvspan(H-S - 0.5, H - 0.5,   color="#e9c46a", alpha=0.15,
                   label=f"fresh (s={S})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Timestep within chunk")
        ax.set_ylabel("Action value (DOF 0 = joints[0])")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")

def eval_prediction_vs_gt(pi, obs_list, model) -> bool:
    """latest addition: predicted action vs dataset GT."""
    print(f"\n=== (5) Prediction vs GT (dataset) ===")
    print(f"    H={H}  seeds={SEEDS[:1]}  obs=dataset ({len(obs_list)} steps)\n")

    errors = []
    for step_idx, (obs, A_prev_np) in enumerate(make_step_iter(obs_list, model)):
        A_pred = run_naive(pi, obs, seed=0)          # (H, MAX_A)
        err    = float(np.mean(np.linalg.norm(A_pred - A_prev_np, axis=-1)))
        errors.append(err)
        print(f"  step={step_idx:03d}  L2={err:.4f}")

    print_stats("\n  mean L2 (pred vs GT)", errors)
    return True
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
        default="~/projects/crossformer/0403_super-night-806/params")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n_steps", type=int, default=10)
    args = parser.parse_args()

    ckpt_path = str(Path(args.checkpoint).expanduser())
    print(f"Loading checkpoint: {ckpt_path}  step={args.step or 'latest'}")
    model = CrossFormerModel.load_pretrained(ckpt_path, step=args.step)
    print("Checkpoint loaded.")

    pi = model.module.bind({"params": model.params}).heads["xflow"]

    obs_list = []
    if args.dataset:
        obs_list = load_obs_list(model, args.dataset, args.n_steps)
    else:
        print("No --dataset provided, using synthetic example_batch obs.\n")

    q3_pass        = eval_inference_quality(pi, obs_list, model)
    q1_pass, _, _  = eval_boundary_jump(pi, obs_list, model)
    q2_pass        = eval_plan_commitment(pi, obs_list, model)
    sweep_results  = eval_delay_sweep(pi, obs_list, model)

    fig_path = Path("figures/eval_rtc_checkpoint.png")
    eval_figure(pi, obs_list, model, fig_path)

    n_sweep_pass = sum(1 for r in sweep_results if r["order_ok"])
    src = f"dataset ({len(obs_list)} steps)" if obs_list else "synthetic"

    if obs_list:
        eval_prediction_vs_gt(pi, obs_list, model)

    print("\n" + "=" * 60)
    print("=== SUMMARY ===")
    print(f"  obs source : {src}")
    print(f"  (1) Plan commitment  (boundary jump)    : {'PASS' if q1_pass else 'FAIL'}")
    print(f"  (2) Jitter test      (frozen stability) : {'PASS' if q2_pass else 'FAIL'}")
    print(f"  (3) Inference quality (region ordering) : {'PASS' if q3_pass else 'FAIL'}")
    print(f"  (4) Figure                              : {fig_path}")
    print(f"  Delay sweep : {n_sweep_pass}/{len(sweep_results)} (d,s) pairs passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
