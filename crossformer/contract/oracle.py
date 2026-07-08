"""Config-derived batch spec oracle.

The post-collate batch's shapes are not static across experiments, but every
dim is a pure function of config: batch size and window from the loader,
horizon from the trajectory transform, action width from the mix's
embodiments, view count from ``MAX_VIEWS``. This module computes that
expectation *without touching data*, so it can be diffed against an observed
batch (via ``crossformer.utils.spec``) as an oracle that is independent of the
pipeline code being tested.

Two tiers of checking:

* **Predicted keys** — the fixed contract (``act`` block, core masks, image,
  timesteps, language, dataset/head bookkeeping). These are fully derivable
  from config; a missing key, extra-dim, or dtype drift is an error.
* **Rule-checked keys** — data-dependent subtrees (``action.*``,
  ``observation.proprio_*``, ``state.*``, ``info.*``, remaining ``mask.*``)
  whose key sets come from the raw datasets rather than config. These are not
  predicted key-by-key (yet — per-dataset raw schemas are the planned
  upgrade), but every leaf must still obey the axis rules: leading dim ``B``,
  ``action.*``/``state.*`` on the ``TH`` axis, ``observation.*`` on the ``TW``
  axis, masks boolean.

Axis names follow ``crossformer.contract.batch.AXES``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from crossformer.embody import MAX_VIEWS
from crossformer.utils.spec import diff, SimpleSpec, spec
from crossformer.utils.tree import flat

NAME_BYTES = 32  # S: fixed-length encoded-string bytes (str2np / _encode_name)
LANG_DIM = 512  # L: language embedding dim


@dataclass(frozen=True)
class Dims:
    """Resolved axis sizes for one run. Every field is derivable from config."""

    B: int  # per-process batch size (loader.batch_size)
    TW: int  # observation window (cfg.window_size)
    TH: int  # action horizon (cfg.data.traj.action_horizon)
    A: int  # padded action dim: max embodiment.action_dim over the mix
    V: int = MAX_VIEWS
    IH: int = 64
    IW: int = 64
    IC: int = 3
    S: int = NAME_BYTES
    L: int = LANG_DIM

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        sources: Sequence[Any],
        *,
        img_size: tuple[int, int] = (64, 64),
    ) -> Dims:
        """Resolve dims from a train config (duck-typed, no cn import).

        ``sources`` are the mix's data sources (e.g. ``Arec.from_name(name)``),
        each carrying ``.embodiment`` and ``.chunk``. TH comes from ``chunk``
        (the stored action-chunk length in the arec data) — in the grain path
        ``traj.action_horizon`` is not what reaches the batch. ``img_size``
        mirrors ``GrainDataFactory.resize``.
        """
        chunks = {s.chunk for s in sources}
        if len(chunks) != 1:
            raise ValueError(f"mix has inconsistent chunk lengths (TH): {sorted(chunks)}")
        return cls(
            B=cfg.data.loader.batch_size,
            TW=cfg.window_size or 1,
            TH=chunks.pop(),
            A=max(s.embodiment.action_dim for s in sources),
            IH=img_size[0],
            IW=img_size[1],
        )


def expected_batch_spec(dims: Dims, heads: Sequence[str] | None = None) -> dict[str, Any]:
    """Predicted spec for the fixed-contract keys of a post-collate batch.

    ``heads`` are the action-head names (defaults to ``HEAD_TO_DATASET`` keys).
    Returns a nested tree of ``SimpleSpec`` leaves; compare against an observed
    batch with ``check_batch``.
    """
    if heads is None:
        from crossformer.data.oxe import HEAD_TO_DATASET

        heads = list(HEAD_TO_DATASET)

    d = dims
    f32, i32, u8, b = np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.uint8), np.dtype(np.bool_)
    head_masks = {h: SimpleSpec((d.B, d.TW), b) for h in heads}
    return {
        "act": {
            "base": SimpleSpec((d.B, d.TH, d.A), f32),
            "id": SimpleSpec((d.B, d.A), i32),
            "view": SimpleSpec((d.B, d.A), i32),
            "embody": SimpleSpec((d.B, d.S), u8),
        },
        "action_head_masks": head_masks,
        "dataset_name": SimpleSpec((d.B, d.S), u8),
        "language_instruction": SimpleSpec((d.B, d.L), f32),
        "mask": {
            "act": SimpleSpec((d.B, d.A), b),
            "view": SimpleSpec((d.B, d.V), b),
            "horizon": SimpleSpec((d.B, d.TW, d.TH), b),
            "timestep_pad_mask": SimpleSpec((d.B, d.TW), b),
            "action_head_masks": dict(head_masks),
        },
        "observation": {
            "image": SimpleSpec((d.B, d.TW, d.V, d.IH, d.IW, d.IC), u8),
            "timestep": SimpleSpec((d.B, d.TW), i32),
            "timestep_pad_mask": SimpleSpec((d.B, d.TW), b),
        },
        "info": {
            "dataset_name": SimpleSpec((d.B, d.S), u8),
            "id": {
                "step": SimpleSpec((d.B, 1), i32),
                "episode": SimpleSpec((d.B, 1), i32),
            },
            "len": SimpleSpec((d.B, 1), i32),
        },
    }


@dataclass
class ContractReport:
    """Outcome of checking one observed batch against the oracle.

    ``errors`` are contract violations (missing/mismatched predicted keys,
    axis-rule failures on extras). ``unmodeled`` lists extra keys that passed
    the axis rules — present in data but not yet predicted from config.
    """

    errors: list[str] = field(default_factory=list)
    unmodeled: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            bullets = "\n".join(f"  - {e}" for e in self.errors)
            raise ValueError(f"batch violates contract ({len(self.errors)} errors):\n{bullets}")


def _rule_check(key: str, s: SimpleSpec, d: Dims) -> str | None:
    """Axis rules for keys the oracle does not predict. None = passes."""
    shape, dtype = tuple(s.shape), np.dtype(s.dtype)
    if not shape or shape[0] != d.B:
        return f"{key}: leading dim {shape[:1]} != B={d.B}"
    if key.startswith("mask.") and dtype != np.bool_:
        return f"{key}: mask dtype {dtype} != bool"
    if key.startswith(("action.", "state.")) and (len(shape) < 2 or shape[1] != d.TH):
        return f"{key}: axis 1 is {shape[1] if len(shape) > 1 else None}, expected TH={d.TH}"
    if key.startswith("observation.") and (len(shape) < 2 or shape[1] != d.TW):
        return f"{key}: axis 1 is {shape[1] if len(shape) > 1 else None}, expected TW={d.TW}"
    if key.startswith("observation.pad_mask_dict.") and dtype != np.bool_:
        return f"{key}: pad mask dtype {dtype} != bool"
    return None


def check_batch(
    batch: Mapping[str, Any],
    dims: Dims,
    heads: Sequence[str] | None = None,
) -> ContractReport:
    """Diff an observed batch against the config-derived oracle.

    Predicted keys must match exactly (shape and dtype). Keys the oracle does
    not predict are held to the axis rules and reported as ``unmodeled``.
    """
    expected = flat(expected_batch_spec(dims, heads))
    observed = spec(flat(dict(batch)))

    report = ContractReport()
    d = diff(expected, observed)

    for key, want in d["removed"].items():
        report.errors.append(f"{key}: missing (expected {tuple(want.shape)} {np.dtype(want.dtype)})")
    for key, change in d["changed"].items():
        want, got = change["from"], change["to"]
        report.errors.append(
            f"{key}: expected {tuple(want.shape)} {np.dtype(want.dtype)}, got {tuple(got.shape)} {np.dtype(got.dtype)}"
        )
    for key, got in d["added"].items():
        if not isinstance(got, SimpleSpec):  # non-array leaf (str, scalar, ...)
            report.unmodeled.append(f"{key} (non-array: {type(got).__name__})")
            continue
        err = _rule_check(str(key), got, dims)
        if err is not None:
            report.errors.append(err)
        else:
            report.unmodeled.append(str(key))
    return report
