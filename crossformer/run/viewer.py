"""Viser-based live viewer for policy inputs and outputs.

``PolicyViewer`` owns a viser server and can display any subset of: camera
images (GUI panel), robot joint values (GUI numbers), and kp3dc keypoint
clouds (3D scene). ``ViserWrappedPolicy`` wraps a policy and pushes each
step's inputs and outputs to a viewer before returning — display errors are
logged, never raised, so visualization can't break serving.

viser arrives transitively via pyroki; this module is only imported when
viewing is enabled.
"""

from __future__ import annotations

import logging

import numpy as np
import viser
from webpolicy.base_policy import BasePolicy

from crossformer.run.wrappers import PolicyWrapper

log = logging.getLogger(__name__)

Images = np.ndarray | list[np.ndarray] | dict[str, np.ndarray]
Points = np.ndarray | dict[str, np.ndarray]

# distinct colors cycled across point-cloud keys
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 80, 80),
    (80, 160, 230),
    (90, 200, 120),
    (240, 180, 60),
    (180, 100, 220),
    (100, 220, 220),
)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    scaled = img * 255 if img.max() <= 1.5 else img
    return scaled.clip(0, 255).astype(np.uint8)


def _named_images(images: Images) -> dict[str, np.ndarray]:
    """Normalize ndarray | list | dict to {name: HWC uint8}, squeezing lead dims."""
    if isinstance(images, dict):
        named = images
    elif isinstance(images, (list, tuple)):
        named = {f"cam{i}": img for i, img in enumerate(images)}
    else:
        arr = np.asarray(images)
        while arr.ndim > 4:
            arr = arr[0]
        named = {"cam0": arr} if arr.ndim == 3 else {f"cam{i}": v for i, v in enumerate(arr)}
    out = {}
    for name, img in named.items():
        img = np.asarray(img)
        while img.ndim > 3:
            img = img[0]
        out[name] = _to_uint8(img)
    return out


class PolicyViewer:
    """Manage a viser server displaying images, joints, and kp3dc clouds.

    All ``show_*`` methods are idempotent per key: GUI handles are created
    once and updated in place; scene nodes are re-added at a stable name
    (viser replaces nodes with the same name).
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.server = viser.ViserServer(host=host, port=port)
        self._images: dict[str, viser.GuiImageHandle] = {}
        self._numbers: dict[str, viser.GuiNumberHandle] = {}
        self._folders: dict[str, viser.GuiFolderHandle] = {}
        self._colors: dict[str, tuple[int, int, int]] = {}

    def show(
        self,
        *,
        images: Images | None = None,
        joints: np.ndarray | None = None,
        kp3dc: Points | None = None,
        prefix: str = "input",
    ) -> None:
        """Display any subset of images / joints / kp3dc under ``prefix``."""
        if images is not None:
            self.show_images(images, prefix=prefix)
        if joints is not None:
            self.show_joints(joints, prefix=prefix)
        if kp3dc is not None:
            self.show_kp3dc(kp3dc, prefix=prefix)

    def show_images(self, images: Images, *, prefix: str = "input") -> None:
        for name, img in _named_images(images).items():
            key = f"{prefix}/{name}"
            if key in self._images:
                self._images[key].image = img
            else:
                with self._folder(prefix):
                    self._images[key] = self.server.gui.add_image(img, label=name)

    def show_joints(self, joints: np.ndarray, *, prefix: str = "input") -> None:
        """Display a 1D joint vector as read-only GUI numbers (extra lead dims take [0])."""
        vec = np.asarray(joints, dtype=np.float32)
        while vec.ndim > 1:
            vec = vec[0]
        for i, v in enumerate(vec):
            key = f"{prefix}/j{i}"
            if key in self._numbers:
                self._numbers[key].value = float(v)
            else:
                with self._folder(prefix):
                    self._numbers[key] = self.server.gui.add_number(f"j{i}", float(v), disabled=True)

    def show_kp3dc(self, kp3dc: Points, *, prefix: str = "input") -> None:
        named = kp3dc if isinstance(kp3dc, dict) else {"kp3dc": kp3dc}
        for name, pts in named.items():
            self.show_points(f"/{prefix}/{name}", np.asarray(pts).reshape(-1, 3))

    def show_points(self, name: str, points: np.ndarray, *, point_size: float = 0.01) -> None:
        if name not in self._colors:
            self._colors[name] = _PALETTE[len(self._colors) % len(_PALETTE)]
        self.server.scene.add_point_cloud(
            name,
            points=np.asarray(points, dtype=np.float32).reshape(-1, 3),
            colors=self._colors[name],
            point_size=point_size,
        )

    def _folder(self, prefix: str) -> viser.GuiFolderHandle:
        if prefix not in self._folders:
            self._folders[prefix] = self.server.gui.add_folder(prefix)
        return self._folders[prefix]


class ViserWrappedPolicy(PolicyWrapper):
    """Plot each step's inputs (obs images, joints, kp3dc) and outputs
    (predicted joints, position chunk) to a PolicyViewer, then return."""

    def __init__(self, inner: BasePolicy, viewer: PolicyViewer) -> None:
        super().__init__(inner)
        self.viewer = viewer

    def step(self, payload: dict, **kwargs: object) -> dict:
        try:
            self._show_inputs(payload)
        except Exception:
            log.exception("viser input display failed")
        result = self.inner.step(payload, **kwargs)
        try:
            self._show_outputs(result)
        except Exception:
            log.exception("viser output display failed")
        return result

    def _show_inputs(self, payload: dict) -> None:
        obs = payload.get("observation", {})
        proprio = obs.get("proprio", {})
        joints = proprio.get("joints")
        gripper = proprio.get("gripper")
        if joints is not None and gripper is not None:
            joints = np.concatenate([np.atleast_1d(np.squeeze(joints)), np.atleast_1d(np.squeeze(gripper))])
        kp3dc = {k: v for k, v in proprio.items() if "kp3dc" in k}
        self.viewer.show(
            images=obs.get("image"),
            joints=joints,
            kp3dc=kp3dc or None,
            prefix="input",
        )

    def _show_outputs(self, result: dict) -> None:
        actions = result.get("actions")
        if not isinstance(actions, dict):
            return
        joints = actions.get("joints")
        if joints is not None:
            first = np.asarray(joints)
            while first.ndim > 2:  # (B, W, H, 7) -> (H, 7)
                first = first[0]
            self.viewer.show_joints(first[0], prefix="output")
        position = actions.get("position")
        if position is not None:  # predicted position chunk as a 3D trajectory
            self.viewer.show_points("/output/position", np.asarray(position).reshape(-1, 3))
