# Plan: Rebuild CrossFormer Data Pipeline with Google Grain

## 1. Objectives and Constraints
- Replace the current TensorFlow/TFDS-based ingestion in `crossformer.data` with a pipeline built on [Google Grain](https://github.com/google/grain) primitives.
- Preserve the high-level API exposed by `crossformer.data.dataset` so model training scripts and configs require minimal changes.
- Support the existing suite of observation, trajectory, and task transforms while leveraging Grain's JAX-first data abstractions.
- Optimize for scalable multi-host training (pjit/pmap) with deterministic sharding, prefetching, and caching semantics compatible with `jax.Array`.
- Maintain compatibility with current dataset mixtures, metadata loading, and statistics utilities (normalization, goal relabeling, etc.).

## 2. Proposed Directory Layout (`crossformer/data/grain`)
```
crossformer/data/grain/
  __init__.py              # high-level API surface that mirrors `dataset.py` entry-points
  builders.py              # dataset specification parsing + Grain Dataset/BatchDataset builders
  pipelines.py             # reusable pipeline compositions (train/eval) with windowing, relabeling, etc.
  transforms.py            # Grain-compatible wrappers for existing trajectory/observation transforms
  threading.py             # thread pool & async helpers replacing `allocate_threads`
  sharding.py              # pjit/pmap sharding utilities and per-host sampling logic
  metadata.py              # statistics loading, caching, and normalization metadata integration
  utils.py                 # shared helpers (key mapping, structure flattening, spec validation)
  tests/
    test_builders.py
    test_pipelines.py
```

## 3. Migration Strategy
1. **Audit Current Pipeline Usage**
   - Inventory public functions/classes imported from `crossformer.data.dataset` by training/inference code.
   - Document the expected `dl.DLataset`-like behaviors (streaming, `traj_map`, filtering, batching, etc.).
   - Map TensorFlow-specific utilities (e.g., `tf.data.AUTOTUNE`, `dataset.filter`) to Grain equivalents.

2. **Establish Core Grain Builders**
   - Design a `DatasetConfig` dataclass capturing source (TFDS, local files, GCS), splits, mixture weights, and preprocessing flags.
   - Implement builders that translate `DatasetConfig` into `grain.Dataset` pipelines using `grain.sources` and `grain.random` for deterministic sampling.
   - Provide adapters for legacy dataset descriptors (e.g., JSON mixture configs) to fill `DatasetConfig` instances.

3. **Reimplement Transform API**
   - Translate existing trajectory transforms (`traj_transforms`) into pure Python/JAX functions that operate on nested dicts/pytrees.
   - Wrap them with Grain's `map`/`flat_map`/`window` operators, ensuring statelessness and compatibility with multi-threaded execution.
   - Ensure observation transforms (`obs_transforms`) can run either in Python or JAX (using `jax.vmap` where beneficial).
   - Move TensorFlow-dependent logic (string tensors, `tf.image`) to NumPy/JAX equivalents or `tensorflow_text` replacements.

4. **Chunking and Windowing**
   - Replace `traj_map(...chunk_act_obs...)` with Grain's windowing primitives, emitting batched `jax.numpy` arrays.
   - Add utilities for overlapping windows, action horizons, and optional override window sizes.
   - Validate that padding, mask generation, and subsampling semantics match the TensorFlow version.

5. **Goal Relabeling & Task Augmentation**
   - Port `goal_relabeling` and `task_augmentation` functions to be framework-agnostic (pure Python/JAX) if they are not already.
   - Integrate them into Grain pipelines via map/filter steps with host-level randomness managed by `jax.random` keys.

6. **Normalization and Statistics**
   - Reuse `get_dataset_statistics` and related utilities, replacing any TensorFlow ops with NumPy/JAX operations.
   - Cache normalization metadata per dataset mixture and expose it alongside pipeline constructors.

7. **Threading, Prefetch, and Sharding**
   - Implement per-host sharding and deterministic mixing via `grain.experimental.distribute` or custom sharding helpers.
   - Provide async prefetch to device using `jax.device_put_sharded` or `jax.experimental.multihost_utils`.
   - Replace `allocate_threads` with a Grain-native threadpool configuration that still honors user-provided parallelism hints.

8. **High-Level Entry Points**
   - Expose `get_dataset`, `get_dataloader`, and `build_dataset_from_config` from `crossformer.data.grain.__init__` matching current signatures.
   - Provide feature flags allowing gradual migration (e.g., `use_grain=True`).

9. **Testing & Validation**
   - Unit tests for builders and pipelines using small fake datasets to ensure chunking, padding, and goal relabeling match expectations.
   - Golden tests comparing output structure/stats of TensorFlow vs Grain pipelines on a shared toy dataset.
   - Performance regression checks (throughput, latency) under representative batch sizes.

## 4. Integration Plan
- Add configuration hooks so training scripts can select the Grain pipeline via config (e.g., `data_backend: "grain"`).
- Update documentation (`docs/`) to describe new backend, installation requirements (`pip install grain[jax]`).
- Provide migration guide for custom datasets pointing to new builder API.
- Once feature-parity is confirmed, deprecate TensorFlow pipeline and schedule removal.

## 5. Open Questions / Risks
- Grain maturity for large-scale trajectory datasets; may require custom sources (e.g., WebDataset-like readers).
- Ensuring compatibility with language-conditioned trajectories where text processing previously relied on TensorFlow ops.
- Managing randomness between Python threads & JAX PRNG to avoid data duplication across devices.
- Need to evaluate how `dlimp` integration changes—either replace with Grain or build adapters.

## 6. Milestones
1. **Week 1:** Audit + scaffolding (`builders.py`, configs, simple TFDS loader -> Grain`).
2. **Week 2:** Port trajectory/observation transforms, chunking, normalization.
3. **Week 3:** Implement sharding, prefetch, and integration hooks; add documentation.
4. **Week 4:** Comprehensive testing, benchmarking, parity validation, and rollout toggles.

