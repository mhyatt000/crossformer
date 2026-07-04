"""Field adapters that clean proprio samples before they reach ``OnlineStats``.

An adapter maps a raw per-step field value (plus access to the full step, for
mask lookups) into a batch of independent samples that ``OnlineStats`` consumes
one row at a time. The default adapter is identity: the whole value is a single
sample, so stats are unchanged.

Multiview fields such as ``kp3dc_robot`` carry a camera-view axis whose length
varies per episode (2-4 cameras). Setting ``agg`` to that axis pools all views
together by emitting each view as its own sample, so a single shared
camera-frame statistic is computed regardless of view count. ``mask`` lists
dotted paths under the step's ``mask`` subtree that gate invalid views/keypoints
out of the stats.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class StatsFieldAdapter:
    """Clean a proprio field into ``(samples, masks)`` for ``OnlineStats``.

    Args:
        agg: axis to treat as the sample/view axis. ``None`` (default) makes the
            whole value one sample -- identity behavior. An int pools that axis
            by emitting each slice as its own sample.
        mask: dotted paths (relative to ``step["mask"]``) to AND together into a
            per-sample validity mask. Leading axes that don't align with the
            sample axis (e.g. a horizon dim) are reduced by taking index 0.
    """

    def __init__(self, agg: int | None = None, mask: Sequence[str] = ()) -> None:
        self.agg = agg
        self.mask_keys = list(mask or [])

    def feature_shape(self, value: np.ndarray) -> tuple[int, ...]:
        """Per-sample shape of the stream (sample/view axis removed when pooling)."""
        shape = tuple(np.asarray(value).shape)
        if self.agg is None:
            return shape
        a = self.agg % len(shape)
        return shape[:a] + shape[a + 1 :]

    def clean(self, value: Any, step: dict) -> tuple[np.ndarray, np.ndarray | None]:
        """Return ``(samples, masks)``; each row of ``samples`` is one sample.

        ``samples`` has shape ``(N, *feature_shape)``. ``masks`` is ``None`` (all
        valid) or a matching bool array gating invalid elements.
        """
        value = np.asarray(value)
        if self.agg is None:
            return value[None], None
        v = np.moveaxis(value, self.agg, 0)  # (V, *feat)
        mask = self._resolve_mask(step, v.shape[0], v.shape[1:])
        return v, mask

    def _resolve_mask(self, step: dict, n: int, feat: tuple[int, ...]) -> np.ndarray | None:
        if not self.mask_keys:
            return None
        masks = step.get("mask", {})
        out = np.ones((n, *feat), dtype=bool)
        for key in self.mask_keys:
            raw = np.asarray(self._get(masks, key), dtype=bool)
            # drop leading axes (e.g. horizon) until the sample axis aligns
            while raw.ndim and raw.shape[0] != n:
                raw = raw[0]
            pad = len(feat) - (raw.ndim - 1)
            raw = raw.reshape(*raw.shape, *([1] * pad))
            out &= np.broadcast_to(raw, (n, *feat))
        return out

    @staticmethod
    def _get(tree: dict, dotted: str) -> Any:
        node: Any = tree
        for part in dotted.split("."):
            node = node[part]
        return node
