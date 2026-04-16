"""Lift 2D keypoints → 3D camera frame via xArm7 FK + PnP.

Throwaway script.  Loads one JSON from robot_vga_100k/, runs FK from
joint angles to get world-frame 3D, solves PnP with pixel_xy + K to
recover the camera pose, then transforms world→camera and checks
against ground-truth camera_xyz / pixel_xy.

Camera convention: data uses Blender/OpenGL (camera looks along -z,
y-up).  OpenCV solvePnP assumes +z forward, y-down.  The conversion
is FLIP = diag(1, -1, -1):  cam_cv = FLIP @ cam_bl.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# rotation / transform helpers
# ---------------------------------------------------------------------------

FLIP = np.diag([1.0, -1.0, -1.0])  # Blender ↔ OpenCV camera convention


def Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def tf(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


pos = lambda T: T[:3, 3].copy()

# ---------------------------------------------------------------------------
# xArm7 FK  (joint origins from xarm7_standalone.urdf)
# ---------------------------------------------------------------------------

_H = np.pi / 2  # π/2

# (xyz, rpy) for each of the 7 revolute joints (all axis=z in child frame)
JOINT_PARAMS = [
    ((0, 0, 0.267), (0, 0, 0)),  # joint1
    ((0, 0, 0), (-_H, 0, 0)),  # joint2
    ((0, -0.293, 0), (_H, 0, 0)),  # joint3
    ((0.0525, 0, 0), (_H, 0, 0)),  # joint4
    ((0.0775, -0.3425, 0), (_H, 0, 0)),  # joint5
    ((0, 0, 0), (_H, 0, 0)),  # joint6
    ((0.076, 0.097, 0), (-_H, 0, 0)),  # joint7
]

# gripper keypoints that are part of the 4-bar linkage (finger tips);
# their open-chain FK diverges from the closed-loop sim positions
_FINGER_KPS = {"gripper_left_finger", "gripper_right_finger"}


def xarm7_fk(q: np.ndarray, g: float = 0.0) -> dict[str, np.ndarray]:
    """Return keypoint world positions.  q = 7 joint angles (rad), g = gripper (rad)."""
    T = np.eye(4)
    kp = {"base": pos(T)}

    for i, (xyz, rpy) in enumerate(JOINT_PARAMS):
        T = T @ tf(xyz, rpy)
        kp[f"joint{i + 1}"] = pos(T)
        T = T @ Rz(q[i])

    # eef = link7 frame origin after joint7 rotation
    kp["eef"] = pos(T)
    Tg = T.copy()  # gripper base = eef frame

    # tcp: fixed (0,0,0.172) from gripper base
    kp["tcp"] = pos(Tg @ tf(xyz=(0, 0, 0.172)))

    # --- gripper sub-chain (drive_joint axis = x) ---
    # drive pivot (left outer knuckle)
    T_d = Tg @ tf(xyz=(0, 0.035, 0.059098))
    kp["gripper_drive"] = pos(T_d)
    T_d = T_d @ Rx(g)

    # left finger (from left_outer_knuckle after drive rotation)
    kp["gripper_left_finger"] = pos(T_d @ tf(xyz=(0, 0.035465, 0.042039)))

    # left inner knuckle (from gripper base, axis x, mimic drive)
    T_li = Tg @ tf(xyz=(0, 0.02, 0.074098))
    kp["gripper_left_inner"] = pos(T_li)

    # right outer knuckle (axis -x ⇒ Rx(-g))
    T_ro = Tg @ tf(xyz=(0, -0.035, 0.059098))
    kp["gripper_right_outer"] = pos(T_ro)
    T_ro = T_ro @ Rx(-g)

    # right finger
    kp["gripper_right_finger"] = pos(T_ro @ tf(xyz=(0, -0.035465, 0.042039)))

    # right inner knuckle (axis -x)
    T_ri = Tg @ tf(xyz=(0, -0.02, 0.074098))
    kp["gripper_right_inner"] = pos(T_ri)

    return kp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_sample(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def gt_kps(data: dict) -> dict[str, dict]:
    """Return {name: {world, cam, px, vis}} from JSON keypoints."""
    out = {}
    for kp in data["keypoints"]:
        out[kp["name"]] = {
            "world": np.array(kp["world_xyz"]),
            "cam": np.array(kp["camera_xyz"]),
            "px": np.array(kp["pixel_xy"]),
            "vis": kp["visible"],
        }
    return out


def intrinsics(data: dict) -> np.ndarray:
    return np.array(data["camera"]["intrinsics"]["K"], dtype=np.float64)


def gt_w2c(data: dict) -> np.ndarray:
    """Ground-truth world→camera 4x4 (stored transposed in JSON, Blender convention)."""
    return np.array(data["camera"]["extrinsics"]["world_to_camera"], dtype=np.float64).T


def project_bl(K: np.ndarray, cam_bl: np.ndarray) -> np.ndarray:
    """Blender camera → pixel.  cam_bl has -z forward, y up."""
    cam_cv = FLIP @ cam_bl
    return (K @ cam_cv)[:2] / cam_cv[2]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    root = Path(__file__).resolve().parents[2] / "robot_vga_100k"
    path = root / "view_0.json"
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])

    data = load_sample(path)
    gt = gt_kps(data)
    K = intrinsics(data)
    W2C = gt_w2c(data)
    R_gt = W2C[:3, :3]  # Blender convention
    t_gt = W2C[:3, 3]

    # -- joint angles (degrees → radians) --
    jd = data["joints"]
    q = np.deg2rad([jd[f"joint{i}"] for i in range(1, 8)])
    g = np.deg2rad(jd["gripper_angle"])

    # -- FK --
    fk = xarm7_fk(q, g)

    names = [kp["name"] for kp in data["keypoints"]]
    print("=" * 72)
    print("1) FK vs ground-truth world_xyz")
    print("-" * 72)
    print(f"  {'name':<25s} {'err (mm)':>10s}")
    fk_errs = {}
    for n in names:
        err = np.linalg.norm(fk[n] - gt[n]["world"]) * 1000
        fk_errs[n] = err
        flag = " ***" if err > 1.0 else ""
        print(f"  {n:<25s} {err:10.3f}{flag}")
    print(f"  {'mean':25s} {np.mean(list(fk_errs.values())):10.3f}")
    print(f"  {'max':25s} {np.max(list(fk_errs.values())):10.3f}")
    if any(e > 1.0 for e in fk_errs.values()):
        print("  (finger error is expected: open-chain FK vs closed-loop 4-bar sim)")

    # -- PnP: recover camera pose from FK 3D + pixel 2D --
    # exclude finger keypoints with known FK error from the PnP solve
    pnp_names = [n for n in names if gt[n]["vis"] and n not in _FINGER_KPS]
    obj_pts = np.array([fk[n] for n in pnp_names], dtype=np.float64)
    img_pts = np.array([gt[n]["px"] for n in pnp_names], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
    assert ok, "solvePnP failed"
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.ravel()

    # convert OpenCV pose → Blender convention
    R_bl = FLIP @ R_cv
    t_bl = FLIP @ t_cv

    # camera position in world = -R^T @ t
    cam_pos_pnp = -R_bl.T @ t_bl
    cam_pos_gt = -R_gt.T @ t_gt
    cam_dist_pnp = np.linalg.norm(cam_pos_pnp)
    cam_dist_gt = np.linalg.norm(cam_pos_gt)

    print()
    print("=" * 72)
    print("2) PnP-solved pose vs ground-truth extrinsics  (Blender convention)")
    print("-" * 72)
    rot_err_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_gt.T @ R_bl) - 1) / 2, -1, 1)))
    trans_err_mm = np.linalg.norm(t_bl - t_gt) * 1000
    print(f"  rotation error:    {rot_err_deg:.4f} deg")
    print(f"  translation error: {trans_err_mm:.3f} mm")

    print()
    print("  --- recovered R (Blender w2c) ---")
    for row in R_bl:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print(f"  t = [{t_bl[0]:+.6f}  {t_bl[1]:+.6f}  {t_bl[2]:+.6f}]")

    print()
    print("  --- ground-truth R (Blender w2c) ---")
    for row in R_gt:
        print(f"    [{row[0]:+.6f}  {row[1]:+.6f}  {row[2]:+.6f}]")
    print(f"  t = [{t_gt[0]:+.6f}  {t_gt[1]:+.6f}  {t_gt[2]:+.6f}]")

    print()
    print("  --- camera position in world (= -R^T @ t) ---")
    print(
        f"  PnP: [{cam_pos_pnp[0]:+.4f}  {cam_pos_pnp[1]:+.4f}  {cam_pos_pnp[2]:+.4f}]"
        f"   dist to origin: {cam_dist_pnp:.4f} m"
    )
    print(
        f"  GT:  [{cam_pos_gt[0]:+.4f}  {cam_pos_gt[1]:+.4f}  {cam_pos_gt[2]:+.4f}]"
        f"   dist to origin: {cam_dist_gt:.4f} m"
    )
    print(f"  Δpos: {np.linalg.norm(cam_pos_pnp - cam_pos_gt) * 1000:.3f} mm")

    # -- lift: world → Blender camera using PnP pose --
    print()
    print("=" * 72)
    print("3) Lifted camera_xyz (Blender) vs ground-truth camera_xyz")
    print("-" * 72)
    print(f"  {'name':<25s} {'err (mm)':>10s}")
    lift_errs = []
    for n in names:
        cam_lifted = R_bl @ fk[n] + t_bl
        cam_gt = gt[n]["cam"]
        err = np.linalg.norm(cam_lifted - cam_gt) * 1000
        lift_errs.append(err)
        flag = " ***" if err > 1.0 else ""
        print(f"  {n:<25s} {err:10.3f}{flag}")
    print(f"  {'mean':25s} {np.mean(lift_errs):10.3f}")
    print(f"  {'max':25s} {np.max(lift_errs):10.3f}")

    # -- reprojection error (using OpenCV convention PnP pose) --
    print()
    print("=" * 72)
    print("4) Reprojection error (px)")
    print("-" * 72)
    print(f"  {'name':<25s} {'err (px)':>10s}")
    vis_names = [n for n in names if gt[n]["vis"]]
    reproj_errs = []
    for n in vis_names:
        cam_cv = R_cv @ fk[n] + t_cv
        proj = K @ cam_cv
        px_reproj = proj[:2] / proj[2]
        err = np.linalg.norm(px_reproj - gt[n]["px"])
        reproj_errs.append(err)
        flag = " ***" if err > 2.0 else ""
        print(f"  {n:<25s} {err:10.3f}{flag}")
    print(f"  {'mean':25s} {np.mean(reproj_errs):10.3f}")
    print(f"  {'max':25s} {np.max(reproj_errs):10.3f}")

    # -- sanity: verify gt projection formula --
    print()
    print("=" * 72)
    print("5) GT camera_xyz → pixel sanity check")
    print("-" * 72)
    for n in ["base", "joint3", "tcp"]:
        px_from_gt = project_bl(K, gt[n]["cam"])
        print(f"  {n:<15s}  gt_px={gt[n]['px']}  proj={px_from_gt}  Δ={np.linalg.norm(px_from_gt - gt[n]['px']):.4f}")

    # -- overlay visualisation (saved next to this script, not in dataset dir) --
    img_path = root / data["image_file"]
    if img_path.exists():
        img = cv2.imread(str(img_path))
        for n in vis_names:
            px_gt = gt[n]["px"].astype(int)
            cv2.circle(img, tuple(px_gt), 4, (0, 255, 0), -1)
            cam_cv_ = R_cv @ fk[n] + t_cv
            proj_ = K @ cam_cv_
            px_rp = (proj_[:2] / proj_[2]).astype(int)
            cv2.circle(img, tuple(px_rp), 6, (0, 0, 255), 1)
            cv2.putText(
                img,
                n.replace("gripper_", "g_"),
                (px_gt[0] + 8, px_gt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1,
            )
        out_path = Path(__file__).resolve().parent / "lift_debug.png"
        cv2.imwrite(str(out_path), img)
        print(f"\n  overlay saved → {out_path}")

    print()


if __name__ == "__main__":
    main()
