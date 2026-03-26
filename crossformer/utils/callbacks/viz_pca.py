from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np

from crossformer.utils.callbacks.viz_base import BaseVizCallback
from crossformer.viz.flow_pca import compute_fk, fit_pca, make_fk_fn, prep_data, render_frames


@dataclass
class PCAVizCallback(BaseVizCallback):
    """Render flow trajectories in joint and URDF FK PCA space (Animated)."""

    fk_link: str = "link_eef"
    _fk_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def __call__(self, batch: Mapping[str, Any]) -> np.ndarray:
        # Extract the base (ground truth) and flow (predictions)
        base = self._select_base(self._get(batch, self.base_key))
        flow = self._select_flow(self._get(batch, self.flow_key))

        t0 = perf_counter()

        # Phase 1: Prep & Subsample
        base_sub, flow_sub = prep_data(base, flow)

        # Phase 2: Forward Kinematics
        if self._fk_fn is None:
            self._fk_fn = make_fk_fn(link=self.fk_link)
        base_fk_xyz, flow_fk_xyz = compute_fk(base_sub, flow_sub, self._fk_fn)

        t1 = perf_counter()
        print(f"[PCAVizCallback] Data Prep & FK Time: {t1 - t0:.3f}s")

        # Phase 3: Fit PCA & Render Frames
        joint_state, fk_state, base_joint_2d, base_fk_2d, joint_lim, fk_lim = fit_pca(base_sub, base_fk_xyz)

        frames = render_frames(
            flow_sub, flow_fk_xyz, joint_state, fk_state, base_joint_2d, base_fk_2d, joint_lim, fk_lim, self.figsize
        )

        t2 = perf_counter()
        print(f"[PCAVizCallback] Render Loop Time: {t2 - t1:.3f}s")

        # Returns array of shape [F, H, W, 3] for MP4/GIF saving
        return frames
