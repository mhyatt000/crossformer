# Grain Data Pipeline API

The `crossformer.data.grain` package provides a Google Grain–based data pipeline
that mirrors the behaviour of the legacy TensorFlow input pipeline while
remaining fully NumPy/JAX compatible.  The module exposes builders for turning
raw trajectory sources into Grain datasets, utilities for applying the standard
CrossFormer trajectory and frame transforms, as well as helpers for dataset
normalisation, sharding, and threading.

The sections below document the key entry points, grouped by the part of the
pipeline they control.

## Configuration: `GrainDatasetConfig`

```python
from crossformer.data.grain import GrainDatasetConfig
```

`GrainDatasetConfig` is a dataclass that describes how a raw data source should
be interpreted before it is fed into the Grain pipeline.【F:crossformer/crossformer/data/grain/builders.py†L43-L89】
The most important fields are:

| Field | Description |
| --- | --- |
| `name` | Identifier for the dataset. Propagated to each frame in `dataset_name`. |
| `source` | Sequence of trajectory dicts, a `grain.RandomAccessDataSource`, or a zero-argument callable that produces one. |
| `standardize_fn` | Optional callable or `ModuleSpec` used to rewrite raw trajectories (e.g. rename keys). |
| `image_obs_keys` / `depth_obs_keys` | Mapping from new observation keys to original keys. `None` inserts empty placeholders. |
| `proprio_obs_keys` / `proprio_obs_dims` | Mapping of proprioception streams and their dimensionality. Required when proprio keys are set. |
| `language_key` | Glob-style pattern pointing to language instruction fields inside the raw trajectory. |
| `action_proprio_normalization_type` | One of `metadata.NormalizationType.NORMAL` or `BOUNDS`. Controls how actions/proprio are normalised. |
| `dataset_statistics` | Optional baked-in statistics (`DatasetStatistics`, JSON path, or mapping). When absent, statistics are computed and cached. |
| `statistics_save_dir` | Directory used to cache computed statistics. Defaults to `~/.cache/crossformer`. |
| `force_recompute_dataset_statistics` | Forces statistics recomputation even if cached values are present. |
| `action_normalization_mask` | Boolean mask specifying which action dimensions should be normalised. |
| `filter_fns` | Sequence of predicates (callables or `ModuleSpec`) evaluated on raw trajectories before restructuring. |
| `skip_norm` / `skip_norm_keys` | Disable all normalisation or skip specific proprio keys. |
| `seed` | Base seed forwarded to deterministic random operations. |

The builder resolves callables supplied via `ModuleSpec`, performs optional
filtering, standardises each trajectory, and reshapes the observations into the
canonical `{observation, task, action}` structure expected by the rest of the
pipeline.【F:crossformer/crossformer/data/grain/builders.py†L94-L205】  Dataset statistics are loaded or computed automatically,【F:crossformer/crossformer/data/grain/builders.py†L175-L205】 ensuring
normalisation metadata is always available for downstream stages.

### Building trajectory datasets

The low-level entry point for materialising trajectory datasets is:

```python
from crossformer.data.grain import builders
traj_ds, stats = builders.build_trajectory_dataset(config)
```

`build_trajectory_dataset` consumes a `GrainDatasetConfig`, yields a
`grain.MapDataset` emitting canonical trajectory dicts, and returns the matching
`metadata.DatasetStatistics`.【F:crossformer/crossformer/data/grain/builders.py†L166-L205】  The resulting dataset contains per-step
observations, padded language/task entries, and action arrays ready for further
processing.

## Trajectory-level transforms

```python
from crossformer.data.grain import apply_trajectory_transforms
```

`apply_trajectory_transforms` mirrors the TensorFlow pipeline by applying a
sequence of deterministic operations to a trajectory-level `grain.MapDataset`.【F:crossformer/crossformer/data/grain/pipelines.py†L29-L118】
The key arguments are:

- `train`: enables training-only behaviour such as dataset repetition and
  optional subsampling.
- `window_size`: history length (number of past observations) to retain when
  chunking trajectories into frames.
- `action_horizon`: number of future actions to expose per frame.
- `override_window_size`: clamps the effective history for late timesteps,
  matching the legacy behaviour around episode boundaries.
- `goal_relabeling_strategy` / `goal_relabeling_kwargs`: configure goal
  relabelling (currently only `"uniform"` is supported).
- `subsample_length`: randomly keeps a subset of transitions during training.
- `skip_unlabeled`: drops trajectories without language instructions.
- `max_action` / `max_proprio`: filter trajectories by absolute value bounds.
- `max_action_dim` / `max_proprio_dim`: pad action/proprio streams to fixed
  dimensionality while recording padding masks.
- `post_chunk_transforms`: list of callables applied after chunking (e.g.
  additional augmentation).
- `head_to_dataset`: mapping from policy head names to dataset names used to
  generate per-head action masks.
- `seed`: controls random filtering and sampling operations.

Internally the function invokes the utilities in `transforms.py` for padding,
chunking, mask generation, optional goal relabelling, and trajectory-to-frame
flattening.【F:crossformer/crossformer/data/grain/pipelines.py†L54-L115】  It returns another `grain.MapDataset` emitting chunked
trajectories that can be flattened into frames or batched directly.

## Frame-level transforms

```python
from crossformer.data.grain import apply_frame_transforms
```

`apply_frame_transforms` operates on an `IterDataset` of frame dicts and applies
simple map transforms to each element.【F:crossformer/crossformer/data/grain/pipelines.py†L136-L146】  Transforms can be supplied as callables
or `ModuleSpec` dictionaries and are evaluated sequentially.

## High-level dataset constructors

```python
from crossformer.data.grain import GrainDataset, make_single_dataset, make_interleaved_dataset
```

- `GrainDataset` is a lightweight dataclass bundling an iterable Grain dataset
  with the statistics object used to build it.【F:crossformer/crossformer/data/grain/pipelines.py†L148-L151】
- `make_single_dataset` is the easiest way to build a frame-level dataset for a
  single configuration. It stitches together trajectory building, trajectory
  transforms, optional frame transforms, shuffling, repetition, and batching in
  one call.【F:crossformer/crossformer/data/grain/pipelines.py†L153-L193】  When `train=True` the dataset repeats indefinitely.
- `make_interleaved_dataset` creates a weighted mixture of multiple
  configurations. Each individual dataset is built via `make_single_dataset`
  (without shuffling or batching); frames are then interleaved according to the
  provided `sample_weights` and optionally shuffled/batched at the mixture
  level.【F:crossformer/crossformer/data/grain/pipelines.py†L195-L260】  Statistics are returned as a dictionary keyed by dataset name.

### Example

```python
from crossformer.data.grain import (
    GrainDatasetConfig,
    make_single_dataset,
)

config = GrainDatasetConfig(
    name="bridge",
    source=lambda: my_random_access_source(),
    image_obs_keys={"rgb": "obs/image"},
    proprio_obs_keys={"state": "obs/proprio"},
    proprio_obs_dims={"state": 18},
    language_key="*language*",
)

dataset = make_single_dataset(
    config,
    train=True,
    traj_transform_kwargs={"window_size": 5, "action_horizon": 2},
    frame_transforms=[{"module": "my_pkg.aug", "name": "jitter", "args": [], "kwargs": {}}],
    shuffle_buffer_size=1024,
    batch_size=256,
    seed=42,
)
for batch in dataset.dataset:
    process(batch)
```

## Transform utilities

Most of the behaviour in `apply_trajectory_transforms` is implemented in
`crossformer.data.grain.transforms`.  These functions operate on the canonical
trajectory/frame dictionaries and can be reused directly when crafting custom
pipelines.【F:crossformer/crossformer/data/grain/transforms.py†L1-L208】【F:crossformer/crossformer/data/grain/transforms.py†L209-L268】

Key helpers include:

- `add_pad_mask_dict(traj)`: computes boolean padding masks for observation and
  task sub-dictionaries, storing them under `pad_mask_dict`.
- `pad_actions_and_proprio(traj, max_action_dim, max_proprio_dim)`: pads action
  and proprio streams to fixed widths and records `action_pad_mask`.
- `chunk_action_and_observation(traj, window_size, action_horizon, override_window_size=None)`: creates overlapping observation histories and
  action windows; also generates `timestep_pad_mask`, goal completion masks, and
  updates `action_pad_mask` to ignore future actions beyond episode termination.
- `add_head_action_mask(traj, head_to_dataset)`: builds boolean masks per policy
  head based on the dataset origin for each frame.【F:crossformer/crossformer/data/grain/transforms.py†L42-L79】
- `subsample(traj, length, rng)`: randomly keeps `length` timesteps from a
  trajectory without replacement.【F:crossformer/crossformer/data/grain/transforms.py†L188-L207】
- `uniform_goal_relabel(traj, rng)`: samples new goal observations uniformly
  from within the same trajectory and merges them into `traj["task"]`.
- `flatten_trajectory(traj)`: converts a chunked trajectory into an iterable of
  per-frame dicts suitable for `IterDataset` consumption.【F:crossformer/crossformer/data/grain/transforms.py†L218-L237】
- `zero_out_future_proprio(traj)`: zeros proprio history entries beyond the
  current timestep.
- `drop_empty_language(traj)`: raises if all language annotations are empty,
  useful when filtering datasets that must provide instructions.
- `maybe_cast_dtype(value, dtype)`: convenience helper for enforcing dtypes in
  custom transforms.

## Metadata utilities

```python
from crossformer.data.grain import metadata
```

The `metadata` module stores dataset statistics and normalisation helpers.【F:crossformer/crossformer/data/grain/metadata.py†L1-L199】
Important classes/functions:

- `ArrayStatistics`: summary statistics (mean/std/max/min/p01/p99) for a single
  array. Provides `to_json`/`from_json` for caching.【F:crossformer/crossformer/data/grain/metadata.py†L15-L51】
- `DatasetStatistics`: groups action statistics, per-key proprio statistics, and
  counts of transitions/trajectories.【F:crossformer/crossformer/data/grain/metadata.py†L54-L85】
- `compute_dataset_statistics(trajectories, proprio_keys, hash_dependencies, save_dir=None, force_recompute=False)`: computes statistics across a
  trajectory collection, caches them to disk, and returns a `DatasetStatistics`
  instance.【F:crossformer/crossformer/data/grain/metadata.py†L89-L167】
- `normalize_action_and_proprio(traj, metadata, normalization_type, proprio_keys, action_mask=None, skip_norm_keys=())`: normalises action and proprio
  streams according to either z-score or min/max bounds while respecting action
  masks and opt-out keys.【F:crossformer/crossformer/data/grain/metadata.py†L169-L214】

## Sharding and threading helpers

```python
from crossformer.data.grain import sharding, threading
```

- `sharding.create_shard_options(shard_count=None, shard_index=None, drop_remainder=False, use_jax_process=False)`: wraps the different ways to produce
  `grain.ShardOptions`, either by specifying explicit shard indices or by using
  the current JAX process info.【F:crossformer/crossformer/data/grain/sharding.py†L1-L23】
- `threading.create_read_options(num_threads=None, prefetch_buffer_size=None)`: constructs `grain.ReadOptions` with overrides for Python thread counts and
  prefetch buffer sizes.【F:crossformer/crossformer/data/grain/threading.py†L1-L15】
- `threading.create_multiprocessing_options(num_workers=None, per_worker_buffer_size=None, enable_profiling=False)`: prepares
  `grain.MultiprocessingOptions` instances suitable for worker-based ingestion.

These helpers make it easy to tune Grain's execution behaviour without directly
instantiating Grain primitives.

## Structural utilities

For completeness, `crossformer.data.grain.utils` exposes lightweight tree
helpers used across the pipeline.【F:crossformer/crossformer/data/grain/utils.py†L1-L104】
They are handy when writing custom transforms:

- `tree_map(fn, tree)` and `tree_merge(*trees)` mirror `tf.nest` semantics.
- `clone_structure(value)` copies nested dictionaries while preserving NumPy
  array views.
- `is_padding(value)` / `to_padding(value)` implement the padding heuristics
  used by `add_pad_mask_dict`.
- `ensure_numpy(value)` coerces inputs to `np.ndarray`.
- `as_dict(mapping)` guarantees a mutable dictionary.

## Putting it all together

The typical workflow is:

1. Describe a dataset with `GrainDatasetConfig` (or multiple configs for a
   mixture).
2. Optionally precompute statistics using `builders.build_trajectory_dataset` or
   `metadata.compute_dataset_statistics`.
3. Use `make_single_dataset`/`make_interleaved_dataset` to obtain iterable frame
   datasets and track the returned statistics for model normalisation.
4. Add additional frame transforms or bespoke `transforms` utilities as needed,
   and configure sharding/threading options for the target training setup.

This modular design keeps CrossFormer's data ingestion expressive while taking
full advantage of Grain's composable primitives.
