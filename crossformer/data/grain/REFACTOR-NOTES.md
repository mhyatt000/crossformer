# Grain Refactoring Analysis: loader.py & GrainDataFactory

**Analysis Date**: 2026-02-13
**Scope**: Recursive codebase analysis for unused functions and files in `loader.py` and `GrainDataFactory`
**Status**: Analysis only - no changes made yet

---

## Executive Summary

- **High-confidence pruning candidates**: 8 items (imports, files, dead code blocks)
- **Medium-confidence candidates**: 3 items (mostly internal-only functions)
- **Critical issue found**: Threading module deleted but tests still import it
- **Architectural confusion**: Two `make_single_dataset()` definitions with different signatures

All **functions in loader.py have callers** and are technically used, but several items are dead code or redundant.

---

## 1. UNUSED FUNCTIONS IN loader.py

### Summary: ALL FUNCTIONS HAVE CALLERS ✓

| Function | Call Location(s) | Type | Notes |
|----------|------------------|------|-------|
| `np2jax()` | loader.py:363 | Internal | Called in `GrainDataFactory.make()` |
| `center_crop()` | loader.py:75 | Internal | Called via `functools.partial` in `mix_precompatibility()` |
| `mix_precompatibility()` | loader.py:254 | Internal | Called in `make_single_dataset()` |
| `mix_compatibility()` | loader.py:303 | Internal | Called in `GrainDataFactory.pad_and_mix()` |
| `make_source_by_mix()` | loader.py:326 | Internal | Called in `GrainDataFactory.make()` |
| `make_single_dataset()` | loader.py:274 | Internal | Called in `GrainDataFactory.source2ds()` |
| `GrainDataFactory.source2ds()` | loader.py:331, 336 | Internal | Called in `GrainDataFactory.make()` |
| `GrainDataFactory.pad_and_mix()` | loader.py:337 | Internal | Called in `GrainDataFactory.make()` |
| `GrainDataFactory.make()` | scripts/finetune.py:146, scripts/debug/data_grain.py:52 | External | Externally called |

**Conclusion**: All functions are used; however, many are only internal and could be refactored.

---

## 2. CRITICAL DISCOVERY: Duplicate `make_single_dataset()` with Different Signatures

### The Problem

**Two versions exist with DIFFERENT signatures:**

**loader.py version (line 206):**
```python
def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
    tfconfig: TransformConfig | None = TransformConfig(),
    shuffle_buffer_size: int | None = None,
    drop_remainder: bool = True,
    seed: int = 0,
) -> GrainDataLoader:
```

**pipelines.py version (line 734):**
```python
def make_single_dataset(
    config: builders.GrainDatasetConfig,
    *,
    train: bool,
    shard_fn: Callable,  # ← REQUIRED parameter
    tfconfig: TransformConfig | None = TransformConfig(),
    shuffle_buffer_size: int | None = None,
    drop_remainder: bool = True,
    seed: int = 0,
) -> GrainDataLoader:
```

### Call Pattern Analysis

- **pipelines.py version**:
  - Exported in `__init__.py` (line 11)
  - Used by external code:
    - tests/integration/test_grain_pipeline.py:72, 113
    - tests/grain/test_pipelines.py:44
    - scripts/debug/visualize_data.py:230
    - scripts/debug/data_grain.py:65
    - scripts/debug/compatibility.py:52

- **loader.py version**:
  - Not exported in `__init__.py`
  - Called only internally by `GrainDataFactory.source2ds()` (line 274)
  - That method itself only called by `GrainDataFactory.make()` (internal)

### Recommendation

**Status: loader.py version is likely dead code**

The loader.py version is shadowed by the pipelines.py version which is the public API. The signatures differ (pipelines requires `shard_fn`), indicating they may have diverged in purpose.

---

## 3. UNUSED/DEAD IMPORTS IN loader.py

### High Confidence - Safe to Remove

#### `transforms` import (line 21)
```python
from crossformer.data.grain import builders, transforms  # ← transforms unused
```

- **Evidence**: Imported as module name but:
  - `TransformConfig` is already imported directly from `pipelines` (line 35)
  - The `transforms` module itself is never used
  - One type annotation uses `transforms.TransformConfig` but could use direct import

- **Impact**: Low - just an unused module import
- **Action**: Remove module import; keep `TransformConfig` direct import

#### `ezdiff` import (line 39)
```python
from crossformer.utils.spec import diff, ezdiff, spec  # ← ezdiff only in comments
```

- **Evidence**:
  - Only appears in commented-out code (lines 292, 308)
  - Never used in active code
  - Comment blocks use `ezdiff(a, b, simple=False)` for debug validation

- **Impact**: Low - just an unused import
- **Action**: Remove from import statement

### All Other Imports: USED ✓

All remaining imports are actively used in the file.

---

## 4. DEAD CODE BLOCKS IN loader.py

### Commented-Out Code

| Lines | Content | Purpose |
|-------|---------|---------|
| 72-73 | `print(spec(...))`, `quit()` | Debug output |
| 106 | `print(k, diff[k].shape)` | Debug output |
| 151-152 | `.map(partial(_postprocess_episode...))`, `.filter(exists)` | Disabled transforms |
| 164 | `ds = cache.CacheByKeyMapDataset(ds)` | Disabled caching |
| 292 | `ezdiff(a, b, simple=False)` | Debug validation |
| 304-308 | Large block with `ezdiff()` and `print()` | Debug validation code |

**Total Dead Lines**: ~10 lines of commented code

**Action**: Safe to remove all commented sections

### TODO Markers

| Line | Message |
|------|---------|
| 117 | `# TODO reduce config scope by only passing cfg.data ?` |
| 194 | `log.warning("TODO move cfg.window_size to cfg.data")` |

**Status**: These are valid TODOs for future work, not dead code. Keep them.

---

## 5. UNUSED IMPORTS IN pipelines.py

### Unreachable Dead Code (pipelines.py:837-857)

After the `return` statement on line 835, lines 837-857 are completely unreachable:

```python
return GrainDataLoader(dataset=ds, statistics=self.stats, config=dconfig)

# ↓ UNREACHABLE CODE BELOW ↓
log.warning("TODO add img dropout")
log.warning("TODO add img or lang dropout")
ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=2)
ds = ds.map(np2jax)
...
return GrainDataLoader(...)  # Duplicate return
```

**Evidence**: The second `return` statement is dead code (unreachable)

**Action**: Remove lines 837-857 entirely

---

## 6. DEAD FILES IN grain/map/ SUBDIRECTORY

### Files That ARE Used

| File | Used In |
|------|---------|
| flatmap.py | scripts/make_dset.py, reformat_arec.py, make_dataset_from_memmap.py |
| window.py | scripts/debug/fpa_head.py, chunk.py, compatibility.py |

---

## 7. STATUS OF grain/util/ SUBDIRECTORY

All files are actively used:
- **remap.py**: Used in loader.py, pipelines.py, __init__.py
- **deco.py**: Used in datasets.py (logbar decorator)
- **mano.py**: Used in scripts/make_dset.py, reformat_arec.py, make_dataset_from_memmap.py

---

## 8. CRITICAL ISSUE: Threading Module

**Git Status**: `D threading.py` (deleted)

**Problem**: Test file still imports it:
```python
# tests/grain/test_runtime_options.py:6
from crossformer.data.grain import threading
```

**Impact**: WILL cause test failures

**Action Required**:
- Either restore the threading.py module
- Or skip/update tests/grain/test_runtime_options.py

---

## 9. COMPREHENSIVE PRUNING CHECKLIST

### Tier 1: High Confidence (Safe to Remove)

- [ ] **Remove import**: `ezdiff` from loader.py:39
- [ ] **Remove import**: `transforms` module from loader.py:21 (keep TransformConfig direct import)
- [ ] **Remove dead code**: Lines 72-73 (print + quit)
- [ ] **Remove dead code**: Line 106 (print statement)
- [ ] **Remove dead code**: Lines 151-152 (commented transforms)
- [ ] **Remove dead code**: Line 164 (commented cache)
- [ ] **Remove dead code**: Line 292 (ezdiff debug)
- [ ] **Remove dead code**: Lines 304-308 (large debug block)
- [ ] **Remove dead code**: Lines 837-857 in pipelines.py (unreachable code after return)
- [ ] **Delete file**: grain/map/cache.py
- [ ] **Delete file**: grain/map/limit.py
- [ ] **Delete file**: grain/map/prefetch.py
- [ ] **Delete file**: grain/map/pack.py

### Tier 2: Medium Confidence (Refactoring Candidates)

- [ ] **Consider removing**: `make_single_dataset()` from loader.py if pipelines.py version is the canonical API
- [ ] **Consolidate**: `np2jax()` duplication (also in pipelines.py:818-819)
- [ ] **Verify**: Which `make_single_dataset()` signature is canonical (loader vs pipelines)

### Tier 3: Critical (Must Fix)

- [ ] **Restore or Skip**: threading.py module (deleted but imported by tests)

---

## 10. IMPACT ANALYSIS

### If Tier 1 is removed:
- **Breakage Risk**: Very low
  - Imports are unused
  - Dead code is commented or unreachable
  - Files have no external callers
- **Lines Removed**: ~20 lines of dead code + 4 files

### If Tier 2 is removed:
- **Breakage Risk**: Medium
  - Requires identifying canonical `make_single_dataset()`
  - May need signature alignment
  - GrainDataFactory may need refactoring

### If Tier 3 is not fixed:
- **Breakage Risk**: High ⚠️
  - Tests will fail immediately
  - Must restore threading.py or skip tests

---

## 11. ARCHITECTURE NOTES

### Current Design Issues

1. **Shadowed Function**: `loader.py:make_single_dataset()` is shadowed by `pipelines.py:make_single_dataset()` with different signature
2. **Unclear Responsibility**: Is `GrainDataFactory` the public API or is the pipelines module?
3. **Code Duplication**: `np2jax()` defined in both loader.py and pipelines.py with identical implementation

### Recommendations for Future Refactoring

1. **Make API Boundary Clear**:
   - If pipelines.py is the public API, move GrainDataFactory there or make it a thin wrapper
   - If loader.py is the factory, make the public API call through it

2. **Consolidate Duplicates**:
   - Pick one canonical location for `np2jax()` and `make_single_dataset()`
   - Make signatures consistent

3. **Remove Dead map/ Functions**:
   - cache.py, limit.py, prefetch.py, pack.py are historical artifacts
   - Safe cleanup opportunity

---

## Next Steps

1. **Review Tier 1 items** - these are safe for immediate cleanup
2. **Investigate Tier 2** - requires architectural decision on API boundaries
3. **Fix Tier 3** - urgent (tests are failing)
4. **Then execute pruning** - once review is complete
