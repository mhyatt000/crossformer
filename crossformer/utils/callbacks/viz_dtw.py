from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from crossformer.utils.callbacks.viz_base import BaseVizCallback
from crossformer.utils.dtw import batch_compute_dtw, compute_dtw_path
from crossformer.viz.dtw_plot import render_dtw_alignment_figure


@dataclass
class DTWVizCallback(BaseVizCallback):
    """Render DTW alignment point-to-point comparisons (Static Plot)."""

    band_radius: int = 15
    joint_idx_to_plot: int = 0  # Which specific DOF to plot on the Y-axis

    def __call__(self, batch: Mapping[str, Any]) -> np.ndarray:
        # Extract the base (ground truth) and flow (predictions)
        base = self._select_base(self._get(batch, self.base_key))
        flow = self._select_flow(self._get(batch, self.flow_key))

        t0 = perf_counter()

        # base is typically [Batch, Horizon, DOF]
        human_demo = base[self.sample_idx]

        # flow is typically [Steps, Batch, Horizon, DOF].
        # We want the final diffusion step [-1] for the target sample.
        robot_pred = flow[-1, self.sample_idx] if flow.ndim >= 4 else flow[self.sample_idx]

        # Add batch dimension back for the JAX vmap: [1, Horizon, DOF]
        h_np = np.expand_dims(human_demo, axis=0)
        r_np = np.expand_dims(robot_pred, axis=0)

        # 1. Compute Matrix via JAX
        D_matrices = batch_compute_dtw(h_np, r_np, self.band_radius)
        D_matrix_cpu = np.array(D_matrices[0])

        # 2. Get Path via CPU traceback
        path = compute_dtw_path(D_matrix_cpu)

        t1 = perf_counter()
        print(f"[DTWVizCallback] JAX Compute Time: {t1 - t0:.3f}s")

        # 3. Extract the isolated 1D trajectories for the specific joint
        human_1d = human_demo[:, self.joint_idx_to_plot]
        robot_1d = robot_pred[:, self.joint_idx_to_plot]

        # 4. Render the figure to a NumPy array
        rgb_frame = render_dtw_alignment_figure(
            human_1d,
            robot_1d,
            path,
            title_a=f"Human Base (Joint {self.joint_idx_to_plot})",
            title_b=f"Robot Pred (Joint {self.joint_idx_to_plot})",
        )

        t2 = perf_counter()
        print(f"[DTWVizCallback] Render Loop Time: {t2 - t1:.3f}s")

        # Wrap in an array of 1 frame so it matches the expected [F, H, W, 3] signature
        return np.expand_dims(rgb_frame, axis=0)
