from __future__ import annotations

import copy
import gc
import os
import pickle
import tempfile

import cv2
import numpy as np
import torch

from portrait.pipeline import create_pipeline
from portrait.utils.camera import get_rotation_matrix
from portrait.utils.crop import paste_back, prepare_paste_back
from portrait.utils.io import load_image_rgb, resize_to_limit
from runtime.core.atomic_components.condition_handler import _mirror_index
from runtime.core.atomic_components.motion_stitch import (
    _fix_exp_for_x_d_info_v2,
    _fix_gaze,
    _mix_s_d_info,
    bin66_to_degree,
    ctrl_motion,
    ctrl_vad,
    fade,
    transform_keypoint,
)
from runtime.stream_pipeline import StreamSDK

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LIP_IDX = [6, 12, 14, 17, 19, 20]
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
STAGE0_TARGET_LIP_OPEN_RATIO = 0.2
STAGE0_LIP_CLOSE_OPEN = 40.0
STAGE0_GRIN = 4.86
LIP_NORMALIZE_THRESHOLD = 0.03


def get_fps(path: str, default: float = 25.0) -> float:
    value = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
    return float(value) if value else float(default)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def resolve_input_path(path: str, fallback_base: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    direct = os.path.abspath(path)
    if os.path.exists(direct):
        return direct
    if fallback_base is not None:
        fallback = os.path.abspath(os.path.join(fallback_base, path))
        if os.path.exists(fallback):
            return fallback
    return direct


def create_liveportrait_pipeline(crop_scale: float, face_idx: int):
    return create_pipeline(crop_scale=crop_scale, face_idx=face_idx)


def destroy_liveportrait_pipeline(pipeline) -> None:
    if pipeline is None:
        return
    del pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reshape_lp_points(value, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (1, 21, 3):
        return arr
    if arr.size != 63:
        raise ValueError(f"Unexpected shape for {field_name}: {arr.shape}")
    return arr.reshape(1, 21, 3)


def load_neutral_motion(path: str) -> dict[str, np.ndarray]:
    neutral_path = resolve_input_path(path, CURRENT_DIR)
    if not os.path.isfile(neutral_path):
        raise FileNotFoundError(f"neutral motion not found: {neutral_path}")
    with open(neutral_path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported neutral motion format")
    source_motion = payload.get("motion")
    if not isinstance(source_motion, dict):
        raise ValueError("Neutral motion is missing the motion record")

    expected_sizes = {
        "scale": 1,
        "R": 9,
        "exp": 63,
        "t": 3,
        "kp": 63,
        "x_s": 63,
    }
    motion = {}
    for key, expected_size in expected_sizes.items():
        if key not in source_motion:
            raise ValueError(f"Neutral motion is missing {key}")
        value = np.asarray(source_motion[key], dtype=np.float32)
        if value.size != expected_size or not np.isfinite(value).all():
            raise ValueError(f"Invalid neutral motion field: {key}")
        motion[key] = value.copy()
    motion["scale"] = motion["scale"].reshape(1, 1)
    motion["R"] = motion["R"].reshape(1, 3, 3)
    motion["exp"] = motion["exp"].reshape(1, 21, 3)
    motion["t"] = motion["t"].reshape(1, 3)
    motion["kp"] = motion["kp"].reshape(1, 21, 3)
    motion["x_s"] = motion["x_s"].reshape(1, 21, 3)
    return motion


def preprocess_source_with_neutral_motion(args, pipeline) -> tuple[str, str | None]:
    if args.neutral_motion is None:
        return args.source_path, None

    if args.source_path.lower().endswith(VIDEO_EXTS):
        raise ValueError("Neutral preprocessing currently supports image source only")

    neutral_expression = load_neutral_motion(args.neutral_motion)["exp"]

    neutral_output_path = args.neutral_output_path
    if neutral_output_path is None:
        preprocess_dir = tempfile.mkdtemp(prefix="neutral_preprocess_")
        neutral_output_path = os.path.join(preprocess_dir, os.path.basename(args.source_path))
    else:
        preprocess_dir = None
        if os.path.isdir(neutral_output_path):
            neutral_output_path = os.path.join(
                neutral_output_path,
                os.path.basename(args.source_path),
            )
    neutral_output_path = os.path.abspath(neutral_output_path)
    ensure_dir(os.path.dirname(neutral_output_path))

    wrapper = pipeline.live_portrait_wrapper
    source_rgb = load_image_rgb(args.source_path)
    source_rgb = resize_to_limit(
        source_rgb,
        wrapper.inference_cfg.source_max_dim,
        wrapper.inference_cfg.source_division,
    )
    source_crop_info = pipeline.cropper.crop_source_image(source_rgb, pipeline.cropper.crop_cfg, face_idx=args.face_idx)
    if source_crop_info is None:
        raise RuntimeError(f"No face detected in source image: {args.source_path}")

    I_s = wrapper.prepare_source(source_crop_info["img_crop_256x256"])
    f_s = wrapper.extract_feature_3d(I_s)
    x_s_info = wrapper.get_kp_info(I_s)
    x_s = wrapper.transform_keypoint(x_s_info)

    x_c_s = x_s_info["kp"]
    delta_new = x_s_info["exp"].clone()
    delta_new[:, LIP_IDX, :] = torch.as_tensor(
        neutral_expression[:, LIP_IDX, :],
        device=delta_new.device,
        dtype=delta_new.dtype,
    )
    scale_new = x_s_info["scale"]
    t_new = x_s_info["t"]
    R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
    x_d_new = scale_new * (x_c_s @ R_s + delta_new) + t_new
    x_d_new = wrapper.stitching(x_s, x_d_new)

    out = wrapper.warp_decode(f_s, x_s, x_d_new)
    out = wrapper.parse_output(out["out"])[0]
    mask_ori = prepare_paste_back(
        wrapper.inference_cfg.mask_crop,
        source_crop_info["M_c2o"],
        dsize=(source_rgb.shape[1], source_rgb.shape[0]),
    )
    out_to_ori_blend = paste_back(out, source_crop_info["M_c2o"], source_rgb, mask_ori)
    cv2.imwrite(neutral_output_path, cv2.cvtColor(out_to_ori_blend, cv2.COLOR_RGB2BGR))

    if args.retarget:
        apply_stage0_neutral_retarget_controls(
            image_path=neutral_output_path,
            crop_scale=args.crop_scale,
            face_idx=args.face_idx,
            target_lip_open_ratio=STAGE0_TARGET_LIP_OPEN_RATIO,
            lip_close_open=STAGE0_LIP_CLOSE_OPEN,
            grin=STAGE0_GRIN,
            pipeline=pipeline,
        )

    return neutral_output_path, preprocess_dir


def apply_stage0_neutral_retarget_controls(
    image_path: str,
    crop_scale: float,
    face_idx: int,
    target_lip_open_ratio: float,
    lip_close_open: float,
    grin: float,
    pipeline,
) -> None:
    pipeline.cropper.crop_cfg.scale = crop_scale
    image_rgb = resize_to_limit(
        load_image_rgb(image_path),
        pipeline.live_portrait_wrapper.inference_cfg.source_max_dim,
        pipeline.live_portrait_wrapper.inference_cfg.source_division,
    )
    crop = pipeline.cropper.crop_source_image(
        image_rgb,
        pipeline.cropper.crop_cfg,
        face_idx=face_idx,
    )
    if crop is None:
        raise RuntimeError(f"No face detected in neutralized source: {image_path}")

    wrapper = pipeline.live_portrait_wrapper
    source_tensor = wrapper.prepare_source(crop["img_crop_256x256"])
    source_features = wrapper.extract_feature_3d(source_tensor)
    source_motion = wrapper.get_kp_info(source_tensor)
    source_points = wrapper.transform_keypoint(source_motion)
    expression = source_motion["exp"].clone()
    grin_value = torch.tensor(grin, dtype=expression.dtype, device=expression.device)
    close_value = torch.tensor(
        lip_close_open,
        dtype=expression.dtype,
        device=expression.device,
    )
    expression[0, 20, 2] -= grin_value * 0.001
    expression[0, 20, 1] -= grin_value * 0.001
    expression[0, 14, 1] -= grin_value * 0.001
    expression[0, 19, 1] += close_value * 0.001
    expression[0, 19, 2] += close_value * 0.0001
    expression[0, 17, 1] -= close_value * 0.0001

    rotation = get_rotation_matrix(
        source_motion["pitch"],
        source_motion["yaw"],
        source_motion["roll"],
    )
    target_points = source_motion["scale"] * (
        source_motion["kp"] @ rotation + expression
    ) + source_motion["t"]
    lip_ratio = wrapper.calc_combined_lip_ratio(
        [[float(target_lip_open_ratio)]],
        crop["lmk_crop"],
    )
    target_points += wrapper.retarget_lip(source_points, lip_ratio)
    target_points = wrapper.stitching(source_points, target_points)
    output = wrapper.warp_decode(source_features, source_points, target_points)
    output = wrapper.parse_output(output["out"])[0]
    mask = prepare_paste_back(
        wrapper.inference_cfg.mask_crop,
        crop["M_c2o"],
        dsize=(image_rgb.shape[1], image_rgb.shape[0]),
    )
    result = paste_back(output, crop["M_c2o"], image_rgb, mask)
    cv2.imwrite(image_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def compute_lip_normalize_delta_kp_from_rgb(
    source_rgb: np.ndarray,
    face_idx: int,
    pipeline,
    source_desc: str = "source",
) -> np.ndarray | None:
    source_rgb = resize_to_limit(
        source_rgb,
        pipeline.live_portrait_wrapper.inference_cfg.source_max_dim,
        pipeline.live_portrait_wrapper.inference_cfg.source_division,
    )
    crop_info = pipeline.cropper.crop_source_image(source_rgb, pipeline.cropper.crop_cfg, face_idx=face_idx)
    if crop_info is None:
        raise RuntimeError(f"No face detected when applying lip normalize on source: {source_desc}")

    wrapper = pipeline.live_portrait_wrapper
    I_s = wrapper.prepare_source(crop_info["img_crop_256x256"])
    x_s_info = wrapper.get_kp_info(I_s)
    x_s = wrapper.transform_keypoint(x_s_info)
    combined_lip_ratio_tensor = wrapper.calc_combined_lip_ratio([[0.0]], crop_info["lmk_crop"])
    if float(combined_lip_ratio_tensor[0][0].item()) < LIP_NORMALIZE_THRESHOLD:
        return None
    lip_delta_kp = wrapper.retarget_lip(x_s, combined_lip_ratio_tensor).detach().cpu().numpy().astype(np.float32)
    return lip_delta_kp


def patch_motion_stitch_for_absolute_lips(
    sdk: StreamSDK,
    lip_normalize_delta_kp: np.ndarray | None = None,
    absolute_lip: bool = False,
    video_s0_lip: bool = False,
    render_exp_frames: list[np.ndarray] | None = None,
) -> None:
    base_motion_stitch = sdk.motion_stitch

    class AbsoluteLipMotionStitchWrapper:
        def __init__(self, base):
            self._base = base

        def __getattr__(self, name):
            return getattr(self._base, name)

        def __call__(self, x_s_info, x_d_info, **kwargs):
            base = self._base
            kwargs = base._merge_kwargs(base.overall_ctrl_info, kwargs)

            if base.scale_ratio is None:
                base.scale_b = x_s_info["scale"].item()
                base.scale_ratio = base.scale_a / base.scale_b
                base._set_scale_ratio(base.scale_ratio)

            # Preserve the absolute generated lip expression during motion refinement.
            abs_lip_exp = reshape_lp_points(x_d_info["exp"], "exp").copy()
            if not hasattr(base, "source_exp0") or base.source_exp0 is None:
                base.source_exp0 = reshape_lp_points(x_s_info["exp"], "exp").copy()

            if base.relative_d and base.d0 is None:
                base.d0 = copy.deepcopy(x_d_info)

            x_d_info = _mix_s_d_info(
                x_s_info,
                x_d_info,
                base.use_d_keys,
                base.d0,
            )

            delta_eye = 0
            if base.drive_eye and base.delta_eye_arr is not None:
                delta_eye = base.delta_eye_arr[
                    base.delta_eye_idx_list[base.idx % len(base.delta_eye_idx_list)]
                ][None]
            x_d_info = _fix_exp_for_x_d_info_v2(
                x_d_info,
                x_s_info,
                delta_eye,
                base.fix_exp_a1,
                base.fix_exp_a2,
                base.fix_exp_a3,
            )

            if kwargs.get("vad_alpha", 1) < 1:
                x_d_info = ctrl_vad(x_d_info, x_s_info, kwargs.get("vad_alpha", 1))

            x_d_info = ctrl_motion(x_d_info, **kwargs)

            if base.fade_type == "d0" and base.fade_dst is None:
                base.fade_dst = copy.deepcopy(x_d_info)

            if "fade_alpha" in kwargs and base.fade_type in ["d0", "s"]:
                fade_alpha = kwargs["fade_alpha"]
                fade_keys = kwargs.get("fade_out_keys", base.fade_out_keys)
                if base.fade_type == "d0":
                    fade_dst = base.fade_dst
                else:
                    if base.fade_dst is not None:
                        fade_dst = base.fade_dst
                    else:
                        fade_dst = copy.deepcopy(x_s_info)
                        if base.is_image_flag:
                            base.fade_dst = fade_dst
                x_d_info = fade(x_d_info, fade_dst, fade_alpha, fade_keys)

            if base.drive_eye:
                if base.pose_s is None:
                    yaw_s = bin66_to_degree(x_s_info["yaw"]).item()
                    pitch_s = bin66_to_degree(x_s_info["pitch"]).item()
                    base.pose_s = [yaw_s, pitch_s]
                x_d_info = _fix_gaze(base.pose_s, x_d_info)

            active_lip_normalize_delta_kp_sequence = getattr(
                sdk,
                "active_lip_normalize_delta_kp_sequence",
                None,
            )
            per_frame_lip_normalize = (
                active_lip_normalize_delta_kp_sequence is not None
            )

            exp_frame = reshape_lp_points(x_d_info["exp"], "exp")
            if absolute_lip:
                # Use the generated lip expression directly.
                exp_frame[0, LIP_IDX] = abs_lip_exp[0, LIP_IDX]
            elif base.relative_d and base.d0 is not None:
                d0_exp = reshape_lp_points(base.d0["exp"], "exp")
                if video_s0_lip and getattr(base, "source_exp0", None) is not None:
                    if per_frame_lip_normalize:
                        # Per-frame Stage0 mode:
                        # S_t + (L_t - L_0) + ΔE_s_t.
                        x_s_exp = reshape_lp_points(x_s_info["exp"], "exp")
                        exp_frame[0, LIP_IDX] = x_s_exp[0, LIP_IDX] + (
                            abs_lip_exp[0, LIP_IDX] - d0_exp[0, LIP_IDX]
                        )
                    else:
                        # Fixed Stage0 mode: relative to source frame zero.
                        exp_frame[0, LIP_IDX] = base.source_exp0[0, LIP_IDX] + (
                            abs_lip_exp[0, LIP_IDX] - d0_exp[0, LIP_IDX]
                        )
                else:
                    # Default mode: relative to current-frame source exp.
                    x_s_exp = reshape_lp_points(x_s_info["exp"], "exp")
                    exp_frame[0, LIP_IDX] = x_s_exp[0, LIP_IDX] + (
                        abs_lip_exp[0, LIP_IDX] - d0_exp[0, LIP_IDX]
                    )

            if per_frame_lip_normalize:
                sequence = active_lip_normalize_delta_kp_sequence
                source_frame_idx = getattr(
                    sdk,
                    "current_source_frame_idx",
                    _mirror_index(base.idx, len(sequence)),
                )
                source_frame_idx = _mirror_index(
                    source_frame_idx,
                    len(sequence),
                )
                active_lip_normalize_delta_kp = np.asarray(
                    sequence[source_frame_idx],
                    dtype=np.float32,
                )
                source_alpha = float(
                    getattr(sdk, "current_source_frame_alpha", 0.0)
                )
                if source_alpha > 1e-9:
                    next_source_frame_idx = _mirror_index(
                        getattr(
                            sdk,
                            "current_source_next_frame_idx",
                            source_frame_idx,
                        ),
                        len(sequence),
                    )
                    next_lip_delta = np.asarray(
                        sequence[next_source_frame_idx],
                        dtype=np.float32,
                    )
                    active_lip_normalize_delta_kp = (
                        active_lip_normalize_delta_kp
                        * (1.0 - source_alpha)
                        + next_lip_delta * source_alpha
                    )
            else:
                active_lip_normalize_delta_kp = getattr(
                    sdk,
                    "active_lip_normalize_delta_kp",
                    lip_normalize_delta_kp,
                )
                if active_lip_normalize_delta_kp is None:
                    active_lip_normalize_delta_kp = lip_normalize_delta_kp

            if active_lip_normalize_delta_kp is not None and (not absolute_lip) and base.relative_d and video_s0_lip:
                scale_arr = np.asarray(x_d_info["scale"], dtype=np.float32)
                if scale_arr.ndim == 0:
                    scale_arr = scale_arr.reshape(1, 1)
                elif scale_arr.ndim == 1:
                    scale_arr = scale_arr.reshape(-1, 1)
                else:
                    scale_arr = scale_arr.reshape(scale_arr.shape[0], -1)
                # Keep scale shape as (B, 1, 1) so it broadcasts with lip delta (B, 21, 3).
                scale_safe = np.maximum(scale_arr[:, :1], 1e-6)[:, :, None]

                lip_delta = np.asarray(
                    active_lip_normalize_delta_kp,
                    dtype=np.float32,
                )
                if lip_delta.ndim == 3 and lip_delta.shape[0] > 1:
                    lip_delta = lip_delta[:1]
                if per_frame_lip_normalize:
                    # Per-frame mode recomputes ΔE_t from that source frame's
                    # normalize ΔK_t and the current driving scale.
                    delta_exp_lip = (lip_delta / scale_safe)[:, LIP_IDX, :]
                else:
                    if not hasattr(base, "delta_exp_lip0") or base.delta_exp_lip0 is None:
                        delta_exp = lip_delta / scale_safe
                        base.delta_exp_lip0 = delta_exp[:, LIP_IDX, :].copy()
                    delta_exp_lip = base.delta_exp_lip0
                exp_frame[0, LIP_IDX] += delta_exp_lip[0]

            x_d_info["exp"] = exp_frame.reshape(1, 63).astype(np.float32)
            if render_exp_frames is not None:
                # Capture the neutral/relative expression consumed by transform_keypoint.
                render_exp_frames.append(x_d_info["exp"].copy())

            if base.x_s is not None:
                x_s = base.x_s
            else:
                x_s = transform_keypoint(x_s_info)
                if base.is_image_flag:
                    base.x_s = x_s

            x_d_pre_stitch = transform_keypoint(x_d_info)
            x_d = x_d_pre_stitch

            if base.flag_stitching:
                x_d = base.stitch_net(x_s, x_d_pre_stitch)

            if active_lip_normalize_delta_kp is not None:
                if (not absolute_lip) and base.relative_d and video_s0_lip:
                    base.idx += 1
                    return x_s, x_d
                lip_delta = np.asarray(
                    active_lip_normalize_delta_kp,
                    dtype=np.float32,
                )
                x_d = x_d.copy()
                x_d[:, LIP_IDX, :] += lip_delta[:, LIP_IDX, :]

            base.idx += 1
            return x_s, x_d

    sdk.motion_stitch = AbsoluteLipMotionStitchWrapper(base_motion_stitch)
