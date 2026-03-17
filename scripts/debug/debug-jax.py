from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from typing import Any


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, out.strip()
    except Exception as exc:  # pragma: no cover - debug script
        return 1, f"{type(exc).__name__}: {exc}"


def _env_dump() -> None:
    print("\n=== Environment ===")
    print(f"python: {sys.version}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.platform()}")
    print(f"cwd: {os.getcwd()}")
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_ENABLE_X64",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "LD_LIBRARY_PATH",
        "PATH",
    ]
    for k in keys:
        v = os.environ.get(k)
        print(f"{k}={v!r}")


def _system_dump() -> None:
    print("\n=== System binaries ===")
    for cmd in (["nvidia-smi"], ["nvcc", "--version"]):
        rc, out = _run_cmd(cmd)
        print(f"$ {' '.join(cmd)} (rc={rc})")
        print(out if out else "<no output>")


def _jax_dump() -> dict[str, Any]:
    print("\n=== JAX import + backend ===")
    import jax
    import jaxlib
    import numpy as np

    info: dict[str, Any] = {
        "jax": jax,
        "jnp": jax.numpy,
        "np": np,
    }
    print(f"jax.__version__={jax.__version__}")
    print(f"jaxlib.__version__={jaxlib.__version__}")
    print(f"default_backend={jax.default_backend()}")
    print(f"process_index={jax.process_index()} process_count={jax.process_count()}")
    print(f"device_count={jax.device_count()} local_device_count={jax.local_device_count()}")

    print("\nDevices:")
    for i, d in enumerate(jax.devices()):
        print(f"  [{i}] {d} platform={d.platform} id={getattr(d, 'id', 'n/a')}")

    # Trigger device transfer + sync early.
    x = jax.numpy.arange(16, dtype=jax.numpy.float32).reshape(4, 4)
    y = x @ x.T
    y.block_until_ready()
    print("warmup matmul: ok")
    return info


def _run_test(name: str, fn) -> bool:
    print(f"\n=== Test: {name} ===")
    try:
        out = fn()
        print(f"{name}: PASS")
        if out is not None:
            print(f"{name} output: {out}")
        return True
    except Exception:
        print(f"{name}: FAIL")
        traceback.print_exc()
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Verbose JAX/CUDA repro for CUDA_ERROR_INVALID_IMAGE")
    ap.add_argument("--size", type=int, default=4096, help="Matrix side length for heavy GEMM test")
    ap.add_argument("--repeat", type=int, default=3, help="Number of repeated calls per test")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--stop-on-fail", action="store_true", help="Exit immediately on first failing test")
    args = ap.parse_args()

    print("### debug-jax.py starting ###")
    _env_dump()
    _system_dump()

    try:
        mods = _jax_dump()
    except Exception:
        print("\nJAX failed during import/backend setup.")
        traceback.print_exc()
        return 2

    jax = mods["jax"]
    jnp = mods["jnp"]
    dtype = jnp.float32 if args.dtype == "float32" else jnp.bfloat16
    results: list[tuple[str, bool]] = []

    @jax.jit
    def test_elemwise(x):
        return jnp.sin(x) + jnp.cos(x) * 2.0

    @jax.jit
    def test_matmul(a, b):
        return a @ b

    @jax.jit
    def test_scan(x):
        def body(c, _):
            y = jnp.tanh(c @ c.T)
            return y, y.mean()

        y, hist = jax.lax.scan(body, x, jnp.arange(8))
        return y, hist[-1]

    def run_elemwise():
        x = jnp.linspace(-10, 10, 2_000_000, dtype=jnp.float32)
        y = None
        for i in range(args.repeat):
            y = test_elemwise(x)
            y.block_until_ready()
            print(f"  iter {i + 1}/{args.repeat}: mean={float(y.mean()):.6f}")
        return {"shape": tuple(y.shape), "dtype": str(y.dtype)}

    def run_matmul():
        n = int(args.size)
        a = jnp.ones((n, n), dtype=dtype)
        b = jnp.full((n, n), 1.001, dtype=dtype)
        c = None
        for i in range(args.repeat):
            c = test_matmul(a, b)
            c.block_until_ready()
            print(f"  iter {i + 1}/{args.repeat}: sum={float(c.sum()):.3e}")
        return {"shape": tuple(c.shape), "dtype": str(c.dtype)}

    def run_scan():
        x = jnp.eye(512, dtype=jnp.float32)
        y = None
        h = None
        for i in range(args.repeat):
            y, h = test_scan(x)
            y.block_until_ready()
            print(f"  iter {i + 1}/{args.repeat}: hist_last={float(h):.6f}")
        return {"shape": tuple(y.shape), "hist_last": float(h)}

    tests = [
        ("jit_elemwise", run_elemwise),
        ("jit_matmul_heavy", run_matmul),
        ("jit_scan_matmul", run_scan),
    ]

    for name, fn in tests:
        ok = _run_test(name, fn)
        results.append((name, ok))
        if args.stop_on_fail and not ok:
            break

    failed = [name for name, ok in results if not ok]
    print("\n=== Summary ===")
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if failed:
        print(f"\nFAILED tests: {failed}")
        return 1
    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
