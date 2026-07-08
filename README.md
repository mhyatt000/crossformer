
# installation

    uv sync

# housekeeping

style of this document is concise

# organization

### config

configuration presets that dont fit cleanly into python-only. might be dynamic depending on data or
they might change frequently

### contract

configuration about agreed upon shapes and keys. only change when adding new embodiment

# rough edges

What the structure tells me

  Duplication across module boundaries. The same names recur in crossformer/ and scripts/: dream.py appears at least 4 times (crossformer/model/dream.py, crossformer/run/dream.py, scripts/train/dream.py, scripts/serve/dream.py), plus _wrappers.py/run/wrappers/, and several viz/synth_viz/rast
  callbacks. That's the classic sign of a research repo where every experiment forked its own entry point instead of parameterizing one.

  Config sprawl. You have cn/ (a config-node tree with dataset/transform/, eval/, model/) and a new crossformer/contract/ and a top-level config/ (untracked) and scripts/configs/. Four places that describe "how a run is set up." Pick one.

  Two data stacks. data/grain/ (with map/, util/) is clearly the current one; data/oxe/, data/arec/, data/mcap.py, data/bpnp.py, data/dba.py, data/dtw.py look like accreted format-specific loaders. The git status confirms churn here (from_zarr.py deleted, make_dset.py deleted).

  Dead/scratch weight in the tree. scripts/old/, wip/, tmp-rast.py, *.bak, tests/broken/, flax_utils.py and types.err.txt at repo root. None of this should be in the tree a newcomer navigates.

  Hot spots that are probably doing too much. utils/spatial/kp.py (105 defs) and data/grain/datasets.py (66) / pipelines.py (50) are large enough to be their own subpackages or to be hiding several concerns.

  How I'd prune if starting over

  1. One entry point per verb, not per experiment. Collapse the dream/robopepp/bela script forks into scripts/train.py, scripts/serve.py, scripts/eval.py, each driven by config. Delete scripts/old/, wip/, tmp-*.py, *.bak, tests/broken/ outright (they're in git history if needed).
  2. Single config system. Keep cn/ or contract/, not both, and fold scripts/configs/ + top-level config/ into it. Right now a reader can't tell which one is authoritative.
  3. One data path. Make data/grain/ the only loader. Move arec/oxe/mcap/zarr conversion into a data/ingest/ (or scripts/data/make/) that produces the grain format, so the training path never imports format-specific code.
  4. Flatten run/ vs scripts/. crossformer/run/ and scripts/ overlap conceptually (both are "how you launch"). Library code (importable, tested) stays in crossformer/; thin argument-parsing shells stay in scripts/. Move logic out of scripts, kill _wrappers.py/wrappers/ duplication.
  5. Break up the giants. spatial/kp.py at 105 defs is a subpackage waiting to happen; grain/datasets.py + pipelines.py are the other candidates.
  6. Root cleanup. flax_utils.py, types.err.txt, tmp-rast.py, multiview.md don't belong at repo root.

  The one-sentence version: the codebase grew by copy-forking experiments; pruning is mostly de-duplicating (one train/serve/eval, one config system, one data path) and deleting the old//wip//broken//bak scaffolding — not rewriting the core model code, which looks reasonably factored
  (model/components/heads/ etc.).
