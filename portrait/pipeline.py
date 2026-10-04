from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config.crop_config import CropConfig
from .config.inference_config import InferenceConfig
from .live_portrait_wrapper import LivePortraitWrapper
from .utils.camera import get_rotation_matrix
from .utils.cropper import Cropper


ROOT = Path(__file__).resolve().parent.parent


class PortraitPipeline:
    def __init__(self, crop_scale: float, face_idx: int):
        runtime = ROOT / "checkpoints" / "runtime"
        inference = InferenceConfig(
            models_config=str(ROOT / "portrait" / "config" / "models.yaml"),
            checkpoint_F=str(runtime / "models" / "appearance_extractor.pth"),
            checkpoint_M=str(runtime / "models" / "motion_extractor.pth"),
            checkpoint_G=str(runtime / "models" / "decoder.pth"),
            checkpoint_W=str(runtime / "models" / "warp_network.pth"),
            checkpoint_S=str(runtime / "models" / "stitch_network.pth"),
            mask_crop_path=str(ROOT / "portrait" / "utils" / "resources" / "mask_template.png"),
        )
        inference.mask_crop = cv2.imread(inference.mask_crop_path, cv2.IMREAD_COLOR)
        crop = CropConfig(
            detector_ckpt_path=str(runtime / "aux_models" / "det_10g.onnx"),
            landmark106_ckpt_path=str(runtime / "aux_models" / "2d106det.onnx"),
            landmark_ckpt_path=str(runtime / "aux_models" / "landmark203.onnx"),
            scale=float(crop_scale),
        )
        self.face_idx = int(face_idx)
        self.live_portrait_wrapper = LivePortraitWrapper(inference_cfg=inference)
        self.cropper = Cropper(crop_cfg=crop)

    def make_motion_template(self, images, eye_ratios, lip_ratios, output_fps=25):
        motions = []
        for image in images:
            info = self.live_portrait_wrapper.get_kp_info(image)
            rotation = get_rotation_matrix(info["pitch"], info["yaw"], info["roll"])
            motions.append(
                {
                    "scale": info["scale"].cpu().numpy().astype(np.float32),
                    "R": rotation.cpu().numpy().astype(np.float32),
                    "exp": info["exp"].cpu().numpy().astype(np.float32),
                    "t": info["t"].cpu().numpy().astype(np.float32),
                    "kp": info["kp"].cpu().numpy().astype(np.float32),
                    "x_s": self.live_portrait_wrapper.transform_keypoint(info)
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                }
            )
        return {
            "n_frames": len(motions),
            "output_fps": float(output_fps),
            "motion": motions,
            "c_eyes_lst": [np.asarray(value, dtype=np.float32) for value in eye_ratios],
            "c_lip_lst": [np.asarray(value, dtype=np.float32) for value in lip_ratios],
        }


def create_pipeline(crop_scale: float, face_idx: int) -> PortraitPipeline:
    return PortraitPipeline(crop_scale, face_idx)
