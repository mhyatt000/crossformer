"""Inspect each sub-source of the xgym_sweep mix.

For each dataset in the mix, prints:
  - top-level step keys
  - step["action"] keys  (-> determines which restructure branch builders.py:147 picks)
  - whether the chosen lang_key exists at the top level (what _restructure_step_mano needs)
  - where any language-ish field lives in the raw step

Run:
    uv run scripts/debug/inspect_sweep_sources.py
    uv run scripts/debug/inspect_sweep_sources.py --mix xgym_sweep
"""

from __future__ import annotations

from dataclasses import dataclass

import tyro

from crossformer.cn.dataset.dataset import Loader
from crossformer.data.grain.loader import make_source_by_mix
from scripts.train.xflow import _resolve_version, make_data_cfg


@dataclass
class Args:
    mix: str = "xgym_sweep"
    xgym_sweep_single_version: str = "0.5.5"
    sweep_mano_version: str | None = None
    batch_size: int = 8


def _find_lang_paths(d, prefix=""):
    hits = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            if "lang" in k.lower():
                hits.append(p)
            hits.extend(_find_lang_paths(v, p))
    return hits


def _pin(name: str, version: str) -> None:
    from crossformer.cn.dataset.mix import DataSource
    from crossformer.data.arec.arec import ArrayRecordBuilder

    ds = DataSource.REGISTRY[name]
    ds.version = version
    ds.builder = ArrayRecordBuilder(name=ds.name, version=version, branch=ds.branch)


def main(args: Args):
    _pin("xgym_sweep_single", args.xgym_sweep_single_version)
    mano_version = args.sweep_mano_version or _resolve_version(None, dataset_name="sweep_mano")
    _pin("sweep_mano", mano_version)
    print(f"pins: xgym_sweep_single={args.xgym_sweep_single_version}  sweep_mano={mano_version}")

    from crossformer.cn.dataset.mix import Arec

    cfg = make_data_cfg(args.mix, args.batch_size, Loader(use_grain=True))
    mix_entries = cfg.data.mix.value.flatten()
    print(f"\nmix {args.mix!r} has {len(mix_entries)} sub-sources: {[m[0] for m in mix_entries]}")

    for name, _ in mix_entries:
        print("\n" + "=" * 72)
        print(f"sub-source: {name}")
        print("=" * 72)
        m = Arec.from_name(name)
        ds, dconfig = make_source_by_mix(m, cfg)

        step = next(iter(ds))
        top_keys = sorted(step.keys())
        print(f"top-level keys     : {top_keys}")

        action = step.get("action", {})
        if isinstance(action, dict):
            print(f"action.keys()      : {sorted(action.keys())}")
            has_k3ds = "k3ds" in action
        else:
            print(f"action (non-dict)  : type={type(action).__name__}")
            has_k3ds = False

        branch = "_restructure_step_mano" if has_k3ds else "_restructure_trajectory"
        print(f"restructure branch : {branch}  (k3ds in action? {has_k3ds})")

        lang_key = dconfig.keys.lang
        print(f"dconfig.keys.lang  : {lang_key!r}")
        if lang_key is not None:
            print(f"lang_key at TOP?   : {lang_key in step}")
        print(f"any 'lang*' paths  : {_find_lang_paths(step)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
