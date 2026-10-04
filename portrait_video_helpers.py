from __future__ import annotations

import torch

import portrait_helpers as base


def render_stage0_neutral_frame(source_rgb, neutral_exp, pipeline, face_idx: int):
    wrapper = pipeline.live_portrait_wrapper
    source_rgb = base.resize_to_limit(
        source_rgb,
        wrapper.inference_cfg.source_max_dim,
        wrapper.inference_cfg.source_division,
    )
    crop = pipeline.cropper.crop_source_image(
        source_rgb,
        pipeline.cropper.crop_cfg,
        face_idx=face_idx,
    )
    if crop is None:
        raise RuntimeError("No face detected in source video frame")

    source_tensor = wrapper.prepare_source(crop["img_crop_256x256"])
    source_features = wrapper.extract_feature_3d(source_tensor)
    source_motion = wrapper.get_kp_info(source_tensor)
    source_points = wrapper.transform_keypoint(source_motion)

    expression = source_motion["exp"].clone()
    neutral_exp = torch.as_tensor(
        neutral_exp,
        device=expression.device,
        dtype=expression.dtype,
    ).reshape(1, 21, 3)
    expression[:, base.LIP_IDX, :] = neutral_exp[:, base.LIP_IDX, :]
    rotation = base.get_rotation_matrix(
        source_motion["pitch"], source_motion["yaw"], source_motion["roll"]
    )
    target_points = source_motion["scale"] * (
        source_motion["kp"] @ rotation + expression
    ) + source_motion["t"]
    target_points = wrapper.stitching(source_points, target_points)

    output = wrapper.warp_decode(source_features, source_points, target_points)
    output = wrapper.parse_output(output["out"])[0]
    mask = base.prepare_paste_back(
        wrapper.inference_cfg.mask_crop,
        crop["M_c2o"],
        dsize=(source_rgb.shape[1], source_rgb.shape[0]),
    )
    return base.paste_back(output, crop["M_c2o"], source_rgb, mask)


def apply_stage0_retarget_to_rgb(
    image_rgb,
    pipeline,
    face_idx: int,
    target_lip_open_ratio: float,
    lip_close_open: float,
    grin: float,
):
    wrapper = pipeline.live_portrait_wrapper
    image_rgb = base.resize_to_limit(
        image_rgb,
        wrapper.inference_cfg.source_max_dim,
        wrapper.inference_cfg.source_division,
    )
    crop = pipeline.cropper.crop_source_image(
        image_rgb,
        pipeline.cropper.crop_cfg,
        face_idx=face_idx,
    )
    if crop is None:
        raise RuntimeError("No face detected in neutralized video frame")

    source_tensor = wrapper.prepare_source(crop["img_crop_256x256"])
    source_features = wrapper.extract_feature_3d(source_tensor)
    source_motion = wrapper.get_kp_info(source_tensor)
    source_points = wrapper.transform_keypoint(source_motion)
    expression = source_motion["exp"].clone()

    grin_value = torch.tensor(grin, dtype=expression.dtype, device=expression.device)
    close_value = torch.tensor(
        lip_close_open, dtype=expression.dtype, device=expression.device
    )
    expression[0, 20, 2] -= grin_value * 0.001
    expression[0, 20, 1] -= grin_value * 0.001
    expression[0, 14, 1] -= grin_value * 0.001
    expression[0, 19, 1] += close_value * 0.001
    expression[0, 19, 2] += close_value * 0.0001
    expression[0, 17, 1] -= close_value * 0.0001

    rotation = base.get_rotation_matrix(
        source_motion["pitch"], source_motion["yaw"], source_motion["roll"]
    )
    target_points = source_motion["scale"] * (
        source_motion["kp"] @ rotation + expression
    ) + source_motion["t"]
    lip_ratio = wrapper.calc_combined_lip_ratio(
        [[float(target_lip_open_ratio)]], crop["lmk_crop"]
    )
    target_points += wrapper.retarget_lip(source_points, lip_ratio)
    target_points = wrapper.stitching(source_points, target_points)

    output = wrapper.warp_decode(source_features, source_points, target_points)
    output = wrapper.parse_output(output["out"])[0]
    mask = base.prepare_paste_back(
        wrapper.inference_cfg.mask_crop,
        crop["M_c2o"],
        dsize=(image_rgb.shape[1], image_rgb.shape[0]),
    )
    return base.paste_back(output, crop["M_c2o"], image_rgb, mask)
