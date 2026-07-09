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

from dataclasses import dataclass, field
from functools import partial
import logging
from pathlib import Path

import numpy as np
import viser
from webpolicy.base_policy import BasePolicy

from crossformer.embody import KP_CHAIN
from crossformer.run.wrappers import PolicyWrapper

log = logging.getLogger(__name__)

# the 14 robot kp3dc chain keypoints, as URDF frame names. link_base..link_tcp
# match the URDF directly; the finger keypoints have no exact link, so they map
# to the nearest gripper links — without them a wrist camera (which mostly sees
# the gripper, with the arm behind it) has too few visible points to solve.
_KP_LINK_MAP: dict[str, str] = {
    "left_finger_joint": "left_outer_knuckle",
    "right_finger_joint": "right_outer_knuckle",
    "left_finger_tip": "left_finger",
    "right_finger_tip": "right_finger",
}
_KP_CHAIN_LINKS: list[str] = [_KP_LINK_MAP.get(n, n) for n in KP_CHAIN]


@dataclass
class ViewerConfig:
    """What the viser viewer renders — see config/serve/viewer.yaml."""

    kp: list[int] | None = None  # keypoint indices to display (None = all)
    horizon: list[int] | None = None  # horizon steps to display (None = all)
    urdf_show: bool = True
    ghost_show: bool = True
    frustum_show: bool | list[int] = True  # True=all cams, False=none, [i,...]=subset
    frustum_distance: float = 0.08  # frustum scale (apex->image depth, m); the image fills the frustum

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ViewerConfig":
        import yaml

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        kp3d = d.get("kp3d") or {}
        urdf = d.get("urdf") or {}
        ghost = urdf.get("ghost") or {}
        cam = (d.get("cam") or {}).get("frustum") or {}

        def _show(x: object) -> bool | list[int]:
            if isinstance(x, str):
                return {"all": True, "none": False}.get(x.lower(), True)
            if isinstance(x, (list, tuple)):
                return [int(i) for i in x]
            return bool(x)

        return cls(
            kp=kp3d.get("kp"),
            horizon=kp3d.get("horizon"),
            urdf_show=bool(urdf.get("show", True)),
            ghost_show=bool(ghost.get("show", True)),
            frustum_show=_show(cam.get("show", True)),
            frustum_distance=float(cam.get("distance", 0.08)),
        )

    def frustum_visible(self, view: int) -> bool:
        if isinstance(self.frustum_show, bool):
            return self.frustum_show
        return int(view) in self.frustum_show


def _sel_indices(indices: list[int] | None, n: int) -> list[int]:
    """Selected indices into [0, n): None -> all; else clamp to range, dedupe, sort."""
    if indices is None:
        return list(range(n))
    return sorted({min(max(int(i), 0), n - 1) for i in indices})

Images = np.ndarray | list[np.ndarray] | dict[str, np.ndarray]
Points = np.ndarray | dict[str, np.ndarray]

# translucent light-blue ghost used to render the final action-horizon pose
_GHOST_COLOR: tuple[int, int, int] = (120, 190, 255)
_GHOST_OPACITY: float = 0.25


class UrdfRobot:
    """Render a URDF in a viser scene at settable joint configs.

    A self-contained stand-in for ``viser.extras.ViserUrdf`` that renders every
    visual mesh via ``add_mesh_simple`` so the caller controls ``color`` and
    ``opacity`` — needed for the translucent ghost, which ViserUrdf can't do
    (it forwards neither, and mesh handles can't set opacity after creation).
    FK wiring mirrors ViserUrdf: a frame per link, meshes as static children of
    their link frame, and ``update_cfg`` moving only the link frames.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        urdf_path: str | Path,
        *,
        root_node_name: str = "/robot",
        color: tuple[int, int, int] | None = None,
        opacity: float | None = None,
    ) -> None:
        import yourdfpy

        path = Path(urdf_path)
        self._urdf = yourdfpy.URDF.load(
            str(path), filename_handler=partial(yourdfpy.filename_handler_magic, dir=str(path.parent))
        )
        self._server = server
        self._root = root_node_name

        # a frame per link (moved by update_cfg); meshes hang statically off them
        self._link_frames: dict[str, viser.SceneNodeHandle] = {}
        for joint in self._urdf.joint_map.values():
            self._link_frames[joint.child] = server.scene.add_frame(self._name(joint.child), show_axes=False)

        self._meshes: list[viser.SceneNodeHandle] = []
        for geom_name, mesh in self._urdf.scene.geometry.items():
            parent = self._urdf.scene.graph.transforms.parents[geom_name]
            m = mesh.copy()
            m.apply_transform(self._urdf.get_transform(geom_name, parent))
            name = self._name(geom_name)
            if color is None and opacity is None:
                self._meshes.append(server.scene.add_mesh_trimesh(name, m))
            else:
                self._meshes.append(
                    server.scene.add_mesh_simple(
                        name,
                        m.vertices,
                        m.faces,
                        color=color if color is not None else (200, 200, 200),
                        opacity=1.0 if opacity is None else opacity,
                    )
                )

    @property
    def num_actuated(self) -> int:
        return len(self._urdf.actuated_joint_names)

    def _name(self, frame_name: str) -> str:
        """Scene-node path from base_frame down to ``frame_name`` (see ViserUrdf)."""
        base = self._urdf.scene.graph.base_frame
        frames: list[str] = []
        while frame_name != base:
            frames.append(frame_name)
            frame_name = self._urdf.scene.graph.transforms.parents[frame_name]
        if self._root != "/":
            frames.append(self._root)
        return "/".join(frames[::-1])

    def _pad_cfg(self, cfg: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.num_actuated, dtype=np.float64)
        cfg = np.asarray(cfg, dtype=np.float64).reshape(-1)
        n = min(cfg.shape[0], vec.shape[0])
        vec[:n] = cfg[:n]
        return vec

    def update_cfg(self, cfg: np.ndarray) -> None:
        """Set actuated joints, padding/truncating ``cfg`` to the URDF's DOF count."""
        import viser.transforms as vtf

        self._urdf.update_cfg(self._pad_cfg(cfg))
        for joint in self._urdf.joint_map.values():
            T = self._urdf.get_transform(joint.child, joint.parent)
            frame = self._link_frames[joint.child]
            frame.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
            frame.position = T[:3, 3]

    def link_positions(self, cfg: np.ndarray, link_names: list[str]) -> np.ndarray:
        """World-frame origins (n, 3) of the named links at joint config ``cfg``.

        NaN rows for names absent from the URDF. Reads FK transforms without
        moving the displayed frames (only ``update_cfg`` does that), so callers
        should pass the same ``cfg`` shown to keep the internal state coherent.
        """
        self._urdf.update_cfg(self._pad_cfg(cfg))
        base = self._urdf.scene.graph.base_frame
        out = np.full((len(link_names), 3), np.nan, dtype=np.float32)
        for i, ln in enumerate(link_names):
            if ln in self._urdf.link_map:
                out[i] = self._urdf.get_transform(ln, base)[:3, 3]
        return out

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


def _project_kp3dc(xyz: np.ndarray, K: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pinhole-project camera-frame points (..., 3) to pixel uv (..., 2); NaN when z<=eps.

    Mirrors crossformer.utils.callbacks.kp3dc_viz.project_kp3dc, inlined so the
    viewer needn't import the wandb-dependent callback module.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    pix = np.einsum("ij,...j->...i", np.asarray(K, dtype=np.float32), xyz)
    z = pix[..., 2:3]
    uv = pix[..., :2] / np.maximum(z, eps)
    return np.where(z > eps, uv, np.nan)


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid 3D-3D alignment: return R (3,3), t (3,) minimizing ||R@src + t - dst||.

    Standard SVD/Kabsch with a reflection guard so R stays a proper rotation.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R.astype(np.float32), (dc - R @ sc).astype(np.float32)


def _c2w_from_w2c(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """c2w 4x4 (parent-from-local, for viser) from a world->camera rotation/translation."""
    R = np.asarray(R, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.T
    T[:3, 3] = (-R.T @ np.asarray(t, dtype=np.float32).reshape(3)).astype(np.float32)
    return T


def _solve_extrinsic(kp_cam: np.ndarray, kp_world: np.ndarray, *, max_resid_m: float = 0.1) -> np.ndarray | None:
    """Camera-to-world (c2w) 4x4 pose by Kabsch-aligning kp3dc to joint FK.

    The model predicts every chain keypoint in the camera frame even when it's
    out of the camera's view, so a rigid 3D-3D fit over all finite
    correspondences recovers the pose directly — no projection, no visibility
    filter, no PnP (which a wrist camera, seeing few keypoints, can't solve).
    Kabsch gives world->camera (``X_cam = R @ X_world + t``); we invert to c2w
    (``R.T``, ``-R.T @ t`` = camera position in world), the parent-from-local
    transform viser expects. None if too few points or the fit isn't rigid
    (median residual > ``max_resid_m``, i.e. the predicted kp don't cohere).
    """
    kp_cam = np.asarray(kp_cam, dtype=np.float32)
    kp_world = np.asarray(kp_world, dtype=np.float32)
    valid = np.all(np.isfinite(kp_cam), axis=1) & np.all(np.isfinite(kp_world), axis=1)
    if int(valid.sum()) < 3:
        return None
    obj, cam = kp_world[valid], kp_cam[valid]
    R, t = _kabsch(obj, cam)  # world->cam (w2c)
    resid = float(np.median(np.linalg.norm((obj @ R.T + t) - cam, axis=1)))
    if resid > max_resid_m:
        return None
    return _c2w_from_w2c(R, t)


def _draw_kp_horizon(
    img: np.ndarray, uv: np.ndarray, *, radius: int = 3, steps: list[int] | None = None, total: int | None = None
) -> None:
    """Draw (S, K, 2) pixel tracks in place, colored red->yellow over the horizon.

    ``uv`` holds only the *selected* horizon rows; ``steps`` are their actual
    horizon indices and ``total`` the last horizon index, so the red->yellow
    color reflects true time (t=step/total) even when steps are subsampled.
    RGB: t=0 -> (255,0,0) red at the start, t=1 -> (255,255,0) yellow at the end.
    """
    import cv2

    S = uv.shape[0]
    steps = list(steps) if steps is not None else list(range(S))
    total = max(total if total is not None else S - 1, 1)

    def _pt(p: np.ndarray) -> tuple[int, int] | None:
        if not np.all(np.isfinite(p)):
            return None
        return int(round(float(p[0]))), int(round(float(p[1])))

    def _color(step: int) -> tuple[int, int, int]:
        return (255, int(255 * step / total), 0)

    for k in range(uv.shape[1]):  # per-keypoint trace between consecutive selected steps
        for s in range(S - 1):
            p0, p1 = _pt(uv[s, k]), _pt(uv[s + 1, k])
            if p0 and p1:
                cv2.line(img, p0, p1, _color(steps[s]), 1)
    for s in range(S):
        c = _color(steps[s])
        for k in range(uv.shape[1]):
            p = _pt(uv[s, k])
            if p:
                cv2.circle(img, p, radius, c, -1)


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

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        *,
        urdf_path: str | Path | None = None,
        show_ghost: bool = True,
    ) -> None:
        self.server = viser.ViserServer(host=host, port=port)
        self._images: dict[str, viser.GuiImageHandle] = {}
        self._numbers: dict[str, viser.GuiNumberHandle] = {}
        self._folders: dict[str, viser.GuiFolderHandle] = {}
        self._colors: dict[str, tuple[int, int, int]] = {}

        # solid robot at the current joints + translucent light-blue ghost at
        # the final action-horizon pose (created only when a URDF is given)
        self._robot: UrdfRobot | None = None
        self._ghost: UrdfRobot | None = None
        if urdf_path is not None:
            self._robot = UrdfRobot(self.server, urdf_path, root_node_name="/robot")
            if show_ghost:
                self._ghost = UrdfRobot(
                    self.server,
                    urdf_path,
                    root_node_name="/ghost",
                    color=_GHOST_COLOR,
                    opacity=_GHOST_OPACITY,
                )

    def show_robot(self, joints: np.ndarray, *, ghost: bool = False) -> None:
        """Set the solid (current) or ghost (final-horizon) robot's joint config."""
        robot = self._ghost if ghost else self._robot
        if robot is None:
            return
        vec = np.asarray(joints, dtype=np.float32)
        while vec.ndim > 1:
            vec = vec[0]
        robot.update_cfg(vec)

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

    def show_kp3dc_on_images(
        self,
        images: Images,
        kp3dc_by_view: dict[int, np.ndarray],
        *,
        focal: float = 515.0,
        orig_hw: tuple[int, int] = (480, 640),
        radius: int = 3,
        prefix: str = "input",
        kp_indices: list[int] | None = None,
        horizon_indices: list[int] | None = None,
    ) -> None:
        """Overlay per-view camera-frame keypoints onto each input view image.

        ``kp3dc_by_view`` maps a 0-based view index (matching image order) to a
        ``(H, n_kp, 3)`` camera-frame trajectory over the action horizon. The
        pipeline squashes the source camera image to a square, so each view is
        first un-squashed back to the original ``orig_hw`` (H, W) before
        projecting with ``K = [[focal,0,W/2],[0,focal,H/2],[0,0,1]]`` — the real
        camera geometry the keypoints were computed in. ``kp_indices`` /
        ``horizon_indices`` restrict which keypoints / horizon steps are drawn
        (None = all). Drawn red->yellow; camera-frame kp are never placed in 3D.
        """
        import cv2

        h, w = orig_hw
        K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float32)
        for v, (name, img) in enumerate(_named_images(images).items()):
            kp = kp3dc_by_view.get(v)
            if kp is None:
                continue
            kp = np.asarray(kp, dtype=np.float32)  # (H, n_kp, 3)
            hsel = _sel_indices(horizon_indices, kp.shape[0])
            ksel = _sel_indices(kp_indices, kp.shape[1])
            sub = kp[np.ix_(hsel, ksel)]  # (len(hsel), len(ksel), 3)
            canvas = cv2.resize(np.ascontiguousarray(img), (w, h))  # cv2 takes (width, height)
            _draw_kp_horizon(canvas, _project_kp3dc(sub, K), radius=radius, steps=hsel, total=kp.shape[0] - 1)
            key = f"{prefix}/kp_{name}"
            if key in self._images:
                self._images[key].image = canvas
            else:
                with self._folder(prefix):
                    self._images[key] = self.server.gui.add_image(canvas, label=f"kp_{name}")

    def fk_link_positions(self, cfg: np.ndarray, link_names: list[str]) -> np.ndarray | None:
        """World-frame FK positions of ``link_names`` at ``cfg`` (None if no URDF)."""
        return None if self._robot is None else self._robot.link_positions(cfg, link_names)

    def show_camera_image(
        self,
        name: str,
        image: np.ndarray,
        c2w: np.ndarray,
        *,
        focal: float = 515.0,
        orig_hw: tuple[int, int] = (480, 640),
        distance: float = 0.08,
    ) -> None:
        """Place ``image`` in the 3D scene as a camera frustum at the c2w pose.

        Uses the frustum's built-in image, so viser draws it exactly filling the
        frustum's image plane — no separate plane to drift out of alignment.
        ``distance`` is the frustum scale (apex->image depth); smaller shrinks
        the frustum and image together. The square model-input image is fit to
        the frustum's ``aspect`` (W/H), un-squashing it. ``fov`` is vertical.
        """
        import viser.transforms as vtf

        h, w = orig_hw
        c2w = np.asarray(c2w, dtype=np.float32)
        self.server.scene.add_camera_frustum(
            name,
            fov=float(2 * np.arctan2(h / 2.0, focal)),
            aspect=float(w) / float(h),
            scale=float(distance),
            image=_to_uint8(np.asarray(image)),
            wxyz=vtf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )

    def show_kp3dc_world(
        self,
        kp3dc_by_view: dict[int, np.ndarray],
        *,
        prefix: str = "world",
        point_size: float = 0.008,
        kp_indices: list[int] | None = None,
        horizon_indices: list[int] | None = None,
    ) -> None:
        """Plot per-view world-frame kp3dc trajectories in the 3D scene, red->yellow.

        ``kp3dc_by_view`` maps view index -> ``(H, n_kp, 3)`` world points (e.g.
        camera-frame kp rigidly aligned to joint FK). Points are colored by true
        horizon step; ``kp_indices`` / ``horizon_indices`` restrict what's drawn.
        """
        for v, kp in kp3dc_by_view.items():
            kp = np.asarray(kp, dtype=np.float32)
            hsel = _sel_indices(horizon_indices, kp.shape[0])
            ksel = _sel_indices(kp_indices, kp.shape[1])
            sub = kp[np.ix_(hsel, ksel)]  # (S, K, 3)
            pts = sub.reshape(-1, 3)
            total = max(kp.shape[0] - 1, 1)
            t = np.repeat(np.array([s / total for s in hsel], dtype=np.float32), sub.shape[1])
            colors = np.stack([np.full(t.shape, 255, np.uint8), (t * 255).astype(np.uint8), np.zeros(t.shape, np.uint8)], axis=1)
            finite = np.all(np.isfinite(pts), axis=1)
            if not finite.any():
                continue
            self.server.scene.add_point_cloud(
                f"/{prefix}/view{v}", points=pts[finite], colors=colors[finite], point_size=point_size
            )

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

    def __init__(
        self,
        inner: BasePolicy,
        viewer: PolicyViewer,
        *,
        kp_focal: float = 515.0,
        kp_orig_hw: tuple[int, int] = (480, 640),
        cfg: ViewerConfig | None = None,
    ) -> None:
        super().__init__(inner)
        self.viewer = viewer
        self.kp_focal = kp_focal
        self.kp_orig_hw = kp_orig_hw
        self.cfg = cfg or ViewerConfig()
        self._last_joints: np.ndarray | None = None  # current [j0..j6, gripper] for FK/PnP

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
        try:
            self._show_kp3dc_images(payload, result)
        except Exception:
            log.exception("viser kp3dc overlay failed")
        if isinstance(result, dict):
            result.pop("viz_image", None)  # internal viz handle; never forward to the client
        return result

    def _show_kp3dc_images(self, payload: dict, result: dict) -> None:
        """Overlay predicted per-view kp3dc onto the model-input images.

        Predicted ``actions['kp3dc_robot']`` is (B, W, H, V, 14, 3); the images
        come from ``result['viz_image']`` (the stacked model-input views exposed
        by GrainlikeWrapper in serve) or ``payload.observation.image`` (debug,
        which feeds the grain_full batch directly).
        """
        actions = result.get("actions") if isinstance(result, dict) else None
        if not isinstance(actions, dict) or "kp3dc_robot" not in actions:
            return
        kp = np.asarray(actions["kp3dc_robot"], dtype=np.float32)
        while kp.ndim > 4:  # (B, W, H, V, 14, 3) -> (H, V, 14, 3): batch 0, last window
            kp = kp[0] if kp.ndim > 5 else kp[-1]
        kp_by_view = {v: kp[:, v] for v in range(kp.shape[1])}

        imgs = result.get("viz_image")
        if imgs is None:
            imgs = payload.get("observation", {}).get("image")
        if imgs is None:
            return
        views = np.asarray(imgs)
        while views.ndim > 4:  # (B, W, V, h, w, C) -> (V, h, w, C): batch 0, last window
            views = views[0] if views.ndim > 5 else views[-1]
        self.viewer.show_kp3dc_on_images(
            views, kp_by_view, focal=self.kp_focal, orig_hw=self.kp_orig_hw,
            kp_indices=self.cfg.kp, horizon_indices=self.cfg.horizon,
        )
        self._show_camera_frustums(views, kp_by_view)

    def _show_camera_frustums(self, views: np.ndarray, kp_by_view: dict[int, np.ndarray]) -> None:
        """Place each enabled view's image in the 3D scene at the camera pose
        recovered via PnP (kp3dc@h0 <-> joint-FK world points)."""
        joints = self._last_joints
        if joints is None or self.cfg.frustum_show is False:
            return
        fk_world = self.viewer.fk_link_positions(joints, _KP_CHAIN_LINKS)
        if fk_world is None:  # no URDF -> no world reference
            return
        for v, (name, img) in enumerate(_named_images(views).items()):
            kp = kp_by_view.get(v)
            if kp is None or not self.cfg.frustum_visible(v):
                continue
            c2w = _solve_extrinsic(kp[0], fk_world)  # Kabsch kp3dc@h0 -> FK world
            if c2w is None:
                continue
            self.viewer.show_camera_image(
                f"/cam/{name}", img, c2w, focal=self.kp_focal, orig_hw=self.kp_orig_hw,
                distance=self.cfg.frustum_distance,
            )

    def _show_inputs(self, payload: dict) -> None:
        obs = payload.get("observation", {})
        # nested proprio, as sent by the robot client (observation.proprio.joints).
        # Callers with flat grain_full obs (observation.proprio_joints) adapt to
        # this shape themselves — see scripts/debug/bela.py.
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
        # solid robot at the current joints (joints already concat with gripper
        # above → [j0..j6, gripper] == the URDF's 8 actuated joints)
        if joints is not None:
            self.viewer.show_robot(joints)
            self._last_joints = np.asarray(joints)  # cached for FK/PnP camera-pose recovery

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
            # ghost robot at the FINAL action-horizon step (a[..., -1, :])
            ghost = first[-1]
            grip = actions.get("gripper")
            if grip is not None:
                g = np.asarray(grip)
                while g.ndim > 2:  # (B, W, H, 1) -> (H, 1)
                    g = g[0]
                ghost = np.concatenate([np.atleast_1d(ghost), np.atleast_1d(g[-1])])
            self.viewer.show_robot(ghost, ghost=True)
        position = actions.get("position")
        if position is not None:  # predicted position chunk as a 3D trajectory
            self.viewer.show_points("/output/position", np.asarray(position).reshape(-1, 3))
