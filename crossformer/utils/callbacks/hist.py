"""Histogram and chunk-plot eval callbacks over bundled actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from crossformer.utils.callbacks.base import EvalContext, getpath
import wandb


@dataclass
class HistCallback:
    """Log denormalized action histograms by DOF for data and predictions."""

    name: str = "hist"
    every: int = 0
    data_key: tuple[str, ...] = ("act", "base")
    dof_key: tuple[str, ...] = ("act", "id")

    def __call__(self, ctx: EvalContext) -> dict[str, Any]:
        data = np.asarray(getpath(ctx.batch, self.data_key))
        dof_ids = getpath(ctx.batch, self.dof_key)
        pred = ctx.pred
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        return {
            "data": self._hist_tree(ctx, data, dof_ids),
            "predict": self._hist_tree(ctx, pred_flat, dof_ids, horizon=data.shape[-2]),
        }

    def _hist_tree(self, ctx: EvalContext, arr: Any, dof_ids: Any, horizon: int | None = None) -> dict[str, Any]:
        vals = ctx.denorm.denormalize(arr, dof_ids, ctx.ds_names, horizon=horizon)
        return {name: wandb.Histogram(xs) for name, xs in vals.items() if xs.size}


@dataclass
class ChunkCallback:
    """Plot denormalized action chunks: ground truth (dashed) vs predicted (solid).

    One subplot per DOF so data vs prediction is easy to compare.
    """

    name: str = "action_chunks"
    every: int = 0
    data_key: tuple[str, ...] = ("act", "base")
    dof_key: tuple[str, ...] = ("act", "id")
    sample_idx: int = 0
    subplot_w: float = 3.5  # width per subplot
    subplot_h: float = 2.5  # height per subplot
    max_cols: int = 4
    dpi: int = 120

    def __call__(self, ctx: EvalContext) -> dict[str, Any]:
        data = np.asarray(getpath(ctx.batch, self.data_key))
        dof_ids = getpath(ctx.batch, self.dof_key)
        pred = ctx.pred
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)

        data_lines = ctx.denorm.sample_lines(data, dof_ids, ctx.ds_names, sample_idx=self.sample_idx)
        pred_lines = ctx.denorm.sample_lines(
            pred_flat,
            dof_ids,
            ctx.ds_names,
            sample_idx=self.sample_idx,
            horizon=data.shape[-2],
        )
        return self._render(data_lines, pred_lines)

    def _render(
        self,
        data_lines: dict[str, np.ndarray],
        pred_lines: dict[str, np.ndarray] | None,
    ) -> dict[str, Any]:
        names = list(data_lines)
        n = len(names)
        if n == 0:
            return {}
        ncols = min(n, self.max_cols)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(self.subplot_w * ncols, self.subplot_h * nrows),
            dpi=self.dpi,
            squeeze=False,
        )
        for i, name in enumerate(names):
            ax = axes[i // ncols, i % ncols]
            h = np.arange(len(data_lines[name]))
            ax.plot(h, data_lines[name], "--", color="C0", label="data")
            if pred_lines is not None and name in pred_lines:
                ax.plot(h, pred_lines[name], "-", color="C1", label="pred")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("H", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)
        # hide unused axes
        for i in range(n, nrows * ncols):
            axes[i // ncols, i % ncols].set_visible(False)
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = buf.reshape((*fig.canvas.get_width_height()[::-1], 4))[..., :3].copy()
        plt.close(fig)

        return {"grid": wandb.Image(frame)}
