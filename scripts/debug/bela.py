"""Drive the BELA/PIO viser viewer from a checkpoint's saved example_batch.

A stand-in for scripts/serve/bela.py while the serve path is being fixed:
instead of starting a webpolicy Server that blocks waiting for a robot
client, this builds the same policy stack and pumps the checkpoint's
``example_batch`` through the viser display in a for-loop. Use it to debug
the ``--viser`` option interactively.

Two modes (both use ``model.example_batch``, sliced one sample per step):

* default (``--no-model``): no inference. Inputs come from example_batch's
  observation; the "output" ghost comes from example_batch's action TARGETS
  (``act.base``/``act.id``), denormalized via ActionDenormWrapper — the same
  denorm path serve/bela.py uses. Fast; isolates the viewer from the model.
* ``--model``: runs the real forward pass (ActionDenormWrapper -> ModelPolicy)
  on each sample so the ghost reflects predictions, not targets.

example_batch is grain_full format (normalized), so proprio is denormalized
before display and it is fed at ``policy.inner`` (the ActionDenormWrapper) —
below the GrainlikeWrapper, which expects raw robot payloads.

Example:
    uv run scripts/debug/bela.py --path ~/bafl/luc-ssl/<run>/params \
        --dataset-name xgym_sweep_single --viser-port 8080
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import jax
import numpy as np
from rich import print
from rich.rule import Rule
import tyro

from crossformer.data.grain import metadata
from crossformer.embody import DOF, KP3DC
from crossformer.run.base_policy import ActionDenormWrapper, slots_to_action_dict
from crossformer.run.policy_factory import Trunk, load_policy
from crossformer.run.viewer import _KP_CHAIN_LINKS, _kabsch  # mapped URDF link names (incl. gripper)
from crossformer.utils.spec import spec

# robot kinematic-chain kp3dc slots: dof id -> index into the 42-dim stats block
_KP3DC_IDX: dict[int, int] = {DOF[n]: i for i, n in enumerate(KP3DC.dof_names)}
_N_KP = len(KP3DC.dof_names) // 3


@dataclass
class Config:
    path: Path = tyro.MISSING  # checkpoint params/ dir
    dataset_name: str = tyro.MISSING  # stats source for action denorm / proprio norm (e.g. xgym_sweep_single)
    step: int | None = None
    trunk: Trunk = "auto"
    head_name: str = "action"
    flow_steps: int | None = None  # None keeps the checkpoint's value
    horizon: int | None = None  # None keeps the checkpoint's value
    use_guidance: bool = False
    guide_keys: tuple[str, ...] = ("action.position", "action.orientation")
    resize_to: int | None = None  # None derives from the checkpoint's trained image size
    host: str = "0.0.0.0"
    viser_port: int = 8080
    steps: int = 0  # number of steps; 0 = loop forever until Ctrl-C
    delay: float = 0.5  # seconds between steps so the viewer is watchable
    unroll_batch: bool = True  # cycle one example_batch sample per step instead of the whole batch
    urdf: Path = Path("./xarm7_standalone.urdf")  # robot shown at current joints + ghost at final-horizon joints
    model: bool = False  # run the model forward pass; default replays example_batch action targets instead
    kp_hw: tuple[int, int] = (480, 640)  # (H, W) original camera size to un-squash views to before projecting kp3d
    kp_focal: float = 515.0  # pinhole focal (fx=fy) for kp3d reprojection; principal point = image center
    viewer_config: Path = Path("./config/serve/viewer.yaml")  # what the viser viewer renders; see the yaml


def _sample(eb: dict, i: int, B: int, unroll: bool) -> dict:
    """Slice example_batch to a single sample (keeping a batch dim of 1)."""
    if not unroll:
        return eb
    j = i % B
    return jax.tree.map(lambda x: x[j : j + 1], eb)


def _view_payload(sample: dict, stats: metadata.DatasetStatistics) -> dict:
    """Adapt a grain_full example_batch sample to the viewer's raw-payload shape.

    ViserWrappedPolicy targets the robot client's payload (nested
    ``observation.proprio.joints`` in real units), so this is the debug-side
    adapter: flat ``observation.proprio_*`` keys -> nested ``proprio`` dict,
    denormalized per-part (example_batch proprio is normalized, grain step 8).
    """
    obs = sample["observation"]
    proprio: dict[str, np.ndarray] = {}
    for k, v in obs.items():
        if not k.startswith("proprio_"):
            continue
        part = k[len("proprio_") :]
        if "kp3d" in part:
            continue  # kp3dc is camera-frame (2D-overlaid) and kp3dw is plotted separately; never 3D here
        arr = np.asarray(v)
        s = stats.proprio.get(part)
        if s is not None:
            try:
                arr = metadata.normalize_arr(arr, stats=s, inv=True)
            except ValueError:  # shape can't broadcast — leave as-is
                pass
        proprio[part] = arr
    return {"observation": {"image": obs.get("image"), "proprio": proprio}}


def _denorm_proprio_kp(arr: np.ndarray, stat: metadata.ArrayStatistics | None) -> np.ndarray:
    if stat is None:
        return arr
    try:
        return metadata.normalize_arr(arr, stats=stat, inv=True)
    except ValueError:  # stats/array shape mismatch — leave normalized
        return arr


def _kp3dc_by_view(sample: dict, stats: metadata.DatasetStatistics) -> dict[int, np.ndarray]:
    """Per-view camera-frame kp3dc from the observation, denormalized (metric cam).

    ``observation.proprio_kp3dc_robot`` is (B, TW, V, K, 3) — already structured
    per view, so no slot gathering. Returns {0-based view index: (TW, K, 3)};
    with TW=1 (inference batch) this is a single frame, not a horizon. Camera-
    frame — projected to 2D by the viewer, never plotted in 3D.
    """
    kc = sample["observation"].get("proprio_kp3dc_robot")
    if kc is None:
        return {}
    arr = np.asarray(kc, dtype=np.float32)
    while arr.ndim > 4:  # (B, TW, V, K, 3) -> (TW, V, K, 3)
        arr = arr[0]
    arr = _denorm_proprio_kp(arr, stats.proprio.get("kp3dc_robot"))
    return {v: arr[:, v] for v in range(arr.shape[1])}


def _kp3dw_world(sample: dict, stats: metadata.DatasetStatistics) -> dict[int, np.ndarray] | None:
    """World-frame kp3dw from the observation, denormalized. {0: (TW, K, 3)}.

    ``proprio_kp3dw_robot`` is already world-frame (common across views), so no
    Kabsch is needed for the observation. Returns None when absent.
    """
    kw = sample["observation"].get("proprio_kp3dw_robot")
    if kw is None:
        return None
    arr = np.asarray(kw, dtype=np.float32)
    while arr.ndim > 3:  # (B, TW, K, 3) -> (TW, K, 3)
        arr = arr[0]
    return {0: _denorm_proprio_kp(arr, stats.proprio.get("kp3dw_robot"))}


def _current_joints(sample: dict, stats: metadata.DatasetStatistics) -> np.ndarray | None:
    """Denormalized [j0..j6, gripper] (8,) from example_batch proprio (last frame)."""
    obs = sample["observation"]

    def denorm(part: str) -> np.ndarray | None:
        v = obs.get(f"proprio_{part}")
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32)
        s = stats.proprio.get(part)
        if s is not None:
            try:
                arr = metadata.normalize_arr(arr, stats=s, inv=True)
            except ValueError:
                pass
        return arr

    j = denorm("joints")
    if j is None:
        return None
    j = j.reshape(-1, 7)[-1]  # last frame's 7 joints
    g = denorm("gripper")
    g = g.reshape(-1)[-1:] if g is not None else np.zeros(1, np.float32)
    return np.concatenate([j, g])


def _kp3dc_world_by_view(cam_by_view: dict[int, np.ndarray], fk_world: np.ndarray) -> dict[int, np.ndarray]:
    """Align each view's camera-frame kp3dc to joint-FK world points via Kabsch.

    Fits a rigid transform from the h=0 keypoints to the FK link positions
    (shared reference -> the views also align with each other), then maps the
    whole horizon to world. Returns {view index: (H, n_kp, 3)} world points.
    """
    fk_valid = np.all(np.isfinite(fk_world), axis=1)
    out: dict[int, np.ndarray] = {}
    for v, cam in cam_by_view.items():
        valid = fk_valid & np.all(np.isfinite(cam[0]), axis=1)
        if valid.sum() < 3:
            continue
        R, t = _kabsch(cam[0][valid], fk_world[valid])
        out[v] = np.einsum("ij,hkj->hki", R, cam) + t
    return out


def _view_images(sample: dict) -> np.ndarray | None:
    """Last-window per-view images (V, h, w, C) from the stacked observation."""
    imgs = sample["observation"].get("image")
    if imgs is None:
        return None
    arr = np.asarray(imgs)
    while arr.ndim > 4:  # (B, W, V, h, w, C) -> take batch 0, last window
        arr = arr[0] if arr.ndim > 5 else arr[-1]
    return arr


def _target_actions(sample: dict, inner: ActionDenormWrapper) -> dict | None:
    """Denormalized action dict from example_batch's padded targets (act.base/id).

    Mirrors ActionDenormWrapper.step: slots -> per-part dict -> inverse-normalize.
    Returns None for inference-only batches (no ``act``).
    """
    act = sample.get("act")
    if not isinstance(act, dict) or "base" not in act or "id" not in act:
        return None
    d = slots_to_action_dict(np.asarray(act["base"]), np.asarray(act["id"]))
    return inner.denorm_new({"actions": d})["actions"]


def _pred_kp3dc_by_view(result: dict) -> dict[int, np.ndarray]:
    """Per-view predicted camera-frame kp3dc over the horizon, from model output.

    ActionDenormWrapper emits ``actions['kp3dc_robot']`` as (B, W, H, V, 14, 3),
    already denormalized. Returns {0-based view index: (H, 14, 3)} for the last
    window step — a real horizon, so the viewer's red->yellow sweep applies.
    """
    acts = result.get("actions")
    if not isinstance(acts, dict) or "kp3dc_robot" not in acts:
        return {}
    kp = np.asarray(acts["kp3dc_robot"], dtype=np.float32)
    while kp.ndim > 4:  # (B, W, H, V, 14, 3) -> (H, V, 14, 3): batch 0, last window
        kp = kp[0] if kp.ndim > 5 else kp[-1]
    return {v: kp[:, v] for v in range(kp.shape[1])}


def main(cfg: Config) -> None:
    print(cfg)
    policy = load_policy(
        cfg.path,
        dataset_name=cfg.dataset_name,
        step=cfg.step,
        trunk=cfg.trunk,
        head_name=cfg.head_name,
        flow_steps=cfg.flow_steps,
        horizon=cfg.horizon,
        use_guidance=cfg.use_guidance,
        guide_keys=cfg.guide_keys,
        resize_to=cfg.resize_to,
    )

    # policy.inner is the ActionDenormWrapper: it consumes grain_full
    # example_batch directly and returns dict-valued 'actions'. The outer
    # GrainlikeWrapper is skipped because it expects *raw* robot payloads.
    inner = policy.inner
    assert isinstance(inner, ActionDenormWrapper), f"unexpected stack; policy.inner={type(inner).__name__}"
    stats = inner.stats

    from crossformer.run.viewer import PolicyViewer, ViewerConfig, ViserWrappedPolicy

    vc = ViewerConfig.from_yaml(cfg.viewer_config) if cfg.viewer_config.exists() else ViewerConfig()
    viewer = PolicyViewer(
        host=cfg.host, port=cfg.viser_port,
        urdf_path=cfg.urdf if vc.urdf_show else None, show_ghost=vc.ghost_show,
    )
    # same overlay params + config as serve so the two render kp3dc identically
    viz = ViserWrappedPolicy(inner, viewer, kp_focal=cfg.kp_focal, kp_orig_hw=cfg.kp_hw, cfg=vc)
    print(f"viser viewer on {cfg.host}:{cfg.viser_port} (urdf={cfg.urdf})")

    model = policy.unwrapped().model
    eb = model.example_batch
    B = jax.tree.leaves(eb["observation"])[0].shape[0]
    head = model.module.bind({"params": model.params}).heads[cfg.head_name]
    print(Rule("example_batch observation spec"))
    print(spec(jax.tree.map(np.asarray, eb["observation"])))
    print(f"batch size B={B}, unroll_batch={cfg.unroll_batch}")
    print(f"head '{cfg.head_name}' max_horizon={head.max_horizon}, horizon override={cfg.horizon}")
    print(f"mode: {'model inference' if cfg.model else 'example_batch action targets'}")

    if not cfg.model and _target_actions(_sample(eb, 0, B, cfg.unroll_batch), inner) is None:
        print("[yellow]warning:[/] example_batch has no 'act' targets; ghost will be empty (use --model)")

    print(Rule("looping example_batch through viser"))
    i = 0
    try:
        while cfg.steps == 0 or i < cfg.steps:
            sample = _sample(eb, i, B, cfg.unroll_batch)

            # inputs: adapt grain_full obs to the viewer's raw-payload shape
            viz._show_inputs(_view_payload(sample, stats))

            # outputs: model prediction (ghost gets predicted joints[-1] of the
            # horizon), or replayed action targets when the batch carries them
            if cfg.model:
                result = inner.step(sample)  # inference + action denorm
            else:
                result = {"actions": _target_actions(sample, inner) or {}}
            viz._show_outputs(result)

            # kp3dc_robot on images: --model routes through the SAME overlay serve
            # uses (predicted horizon in result); --no-model uses the observation's
            # single current frame (this batch has no predictions to sweep)
            if cfg.model:
                viz._show_kp3dc_images(sample, result)
                kp_by_view = _pred_kp3dc_by_view(result)
            else:
                kp_by_view = _kp3dc_by_view(sample, stats)
                imgs = _view_images(sample)
                if kp_by_view and imgs is not None:
                    viewer.show_kp3dc_on_images(
                        imgs, kp_by_view, focal=cfg.kp_focal, orig_hw=cfg.kp_hw,
                        kp_indices=vc.kp, horizon_indices=vc.horizon,
                    )

            # kp3d world: predicted kp has no world frame -> Kabsch-align cam->joint FK;
            # the observation provides kp3dw directly (already world), so prefer it
            world = None if cfg.model else _kp3dw_world(sample, stats)
            if world is None and kp_by_view:
                q = _current_joints(sample, stats)
                fk_world = viewer.fk_link_positions(q, _KP_CHAIN_LINKS) if q is not None else None
                if fk_world is not None:
                    world = _kp3dc_world_by_view(kp_by_view, fk_world)
            if world:
                viewer.show_kp3dc_world(world, kp_indices=vc.kp, horizon_indices=vc.horizon)

            actions = result.get("actions", {})
            shapes = {k: np.asarray(v).shape for k, v in actions.items()} if isinstance(actions, dict) else actions
            print(f"step {i} (sample {i % B}/{B}) -> actions: {shapes}")
            i += 1
            time.sleep(cfg.delay)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main(tyro.cli(Config))
