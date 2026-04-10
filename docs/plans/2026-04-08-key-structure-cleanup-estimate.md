# Key Structure Cleanup Estimate

## Goal

Reduce clunky key naming around:

- `image_primary` / `image_left_wrist` / `proprio_single`
- `pad_mask_dict`

The core issue is that structure is encoded in strings rather than the data
shape itself. `image_x` and `proprio_x` each combine:

- modality
- semantic slot

Likewise, `pad_mask_dict` is technically correct, but not very informative.

## Options

### 1. Minimal rename only

Keep flat payloads. Only rename `pad_mask_dict` to something better such as
`mask` or `valid`.

Examples:

```python
observations = {
    "image_primary": ...,
    "proprio_single": ...,
    "mask": {
        "image_primary": ...,
        "proprio_single": ...,
    },
}
```

Estimate:

- `80-180` changed lines

Why:

- broad usage of `pad_mask_dict`
- most edits are one-line renames
- limited API fallout

### 2. Medium cleanup

Keep flat payloads internally, but stop forcing model code to refer to
`image_primary` and `proprio_single`.

Examples:

```python
ImageTokenizer(obs_keys=["primary"])
LowdimObsTokenizer(obs_keys=["single"])
```

The tokenizer would know which modality subtree or prefix it consumes.

Estimate:

- `150-350` changed lines

Why:

- touches tokenizer APIs
- touches model factory and config construction
- updates tests and some helper code
- avoids a full pipeline rewrite

### 3. Full structural cleanup

Make the data shape explicit rather than encoding structure in strings.

Examples:

```python
observations = {
    "image": {
        "primary": ...,
        "left_wrist": ...,
    },
    "proprio": {
        "single": ...,
    },
    "mask": {
        "image": {
            "primary": ...,
            "left_wrist": ...,
        },
        "proprio": {
            "single": ...,
        },
    },
}
```

Estimate:

- `400-900` changed lines

Why:

- touches dataset creation, transforms, tokenizers, wrappers, normalization,
  task augmentation, tests, docs, and debug scripts
- many files would need small compatibility edits even if the core logic change
  is modest

## Recommendation

If the goal is to make the API less clunky without rewriting the whole stack,
the medium cleanup is the best target.

Working estimate:

- `200-300` changed lines

This keeps the blast radius manageable while improving model-facing code.

## Suggested phases

### Phase 1

Rename `pad_mask_dict` to `mask` or `valid`.

Estimate:

- about `100` lines

### Phase 2

Make tokenizer config stop depending on prefixed flat names.

Examples:

```python
ImageTokenizer(obs_keys=["primary"])
LowdimObsTokenizer(obs_keys=["single"])
```

Estimate:

- about `100-200` additional lines

### Phase 3

If still worth it, migrate to explicit nested modality trees.

Estimate:

- about `300-600` additional lines beyond phases 1-2

## Notes

- These are rough estimates from usage counts and likely touch points, not a
  line-by-line audit.
- The largest uncertainty is backward compatibility.
- Costs increase meaningfully if Grain and non-Grain paths must both support
  old and new layouts at the same time.
