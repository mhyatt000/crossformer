from __future__ import annotations

import crossformer.cn as cn
from crossformer.cn.base import default
from crossformer.data.geometry import (
    _denormalize_kp2d,
    _denormalize_kp2d_np,
    _normalize_kp2d,
    _normalize_kp2d_np,
    _opencv_w2c_np,
    _shrink_crop_image_np,
    _shrink_crop_intrinsics_np,
    _shrink_crop_keypoints_np,
    _shrink_crop_resolution,
    denormalize_kp2d,
    denormalize_kp2d_np,
    normalize_kp2d,
    normalize_kp2d_np,
    opencv_w2c_np,
    shrink_crop_image_np,
    shrink_crop_intrinsics_np,
    shrink_crop_keypoints_np,
    shrink_crop_resolution,
)
from crossformer.utils.callbacks.synth_viz import solve_pnp
from crossformer.utils.imaug import kp_render_mask as _kp_render_mask
from crossformer.utils.imaug import plot_image as _plot_image
from crossformer.utils.imaug import rotate_image_np as _rotate_image_np
from crossformer.utils.imaug import rotate_keypoints_np as _rotate_keypoints_np
from crossformer.utils.imaug import translate_image_np as _translate_image_np
from crossformer.utils.imaug import zoom_image_np as _zoom_image_np

from .config import (
    _checkpoint_state,
    _save_path,
    ADD_THRESHOLDS_MM,
    Config,
    DreamCheckpointModel,
    DreamCheckpointState,
    DreamVizConfig,
    KP_CONF_THRESHOLD,
    KP_MISSING_VALUE,
    KP_PEAK_AMBIGUITY_GAP,
    KP_PEAK_THRESHOLD,
    KP_SMOOTH_RADIUS,
    KP_SMOOTH_SIGMA,
    Optim,
    SOURCE_REAL,
    SOURCE_SYNTH,
)
from .data import (
    add_bg,
    make_coco_dataset,
    make_dataset,
    make_irl_dataset,
    make_shard_fn,
    mix_with_bg,
    prepare_irl_sample_np,
    prepare_sample_np,
)
from .losses import belief_sigma, build_heatmaps, dream_loss_fn, focal_heatmap_loss, mask_loss, mask_target
from .main import main
from .metrics import (
    _mask_iou,
    _pnp_reproj_err,
    _solve_pose_one,
    extract_keypoints,
    keypoint_metrics,
    PNP_MASK_IOU_THRESH,
    PNP_MIN_VALID_KP,
    PNP_REPROJ_THRESH,
    pose_metrics,
    pose_metrics_irl,
)
from .modeling import (
    _count_params,
    _count_trainable_params,
    _image_to_float,
    frozen_keys,
    load_tips_params,
    make_model,
    net_out_size,
)
from .session_calibration import (
    build_multiframe_correspondences,
    calibrate_session_cameras,
    CameraCalibrationResult,
    decode_camera_predictions_for_session,
    FramePrediction,
    get_camera_keys,
    MultiFrameCorrespondences,
    MultiFramePnPConfig,
    RobustStackedPnPResult,
    score_camera_calibration,
    select_diverse_reliable_frames,
    SessionCalibrationConfig,
    SessionCalibrationResult,
    solve_camera_extrinsics_from_frames,
    solve_robust_stacked_pnp,
    summarize_session_calibration,
)
from .train_steps import (
    final_pred_heatmaps,
    make_eval_step_dream,
    make_train_step_dream,
    predict_heatmap_out,
    prepare_pred_heatmaps,
    prepare_pred_mask,
    resize_pred_heatmaps,
    resize_pred_mask,
)
from .viz import _image_u8, maybe_log_viz

__all__ = [name for name in globals() if not name.startswith("__")]
