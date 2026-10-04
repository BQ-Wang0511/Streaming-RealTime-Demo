import numpy as np
import cv2

from ..aux_models.insightface_det import InsightFaceDet
from ..aux_models.insightface_landmark106 import Landmark106
from ..aux_models.landmark203 import Landmark203
from ..aux_models.mediapipe_landmark478 import Landmark478
from ..models.appearance_extractor import AppearanceExtractor
from ..models.motion_extractor import MotionExtractor

from ..utils.crop import crop_image
from ..utils.eye_info import EyeAttrUtilsByMP


class Source2Info:
    def __init__(
        self,
        insightface_det_cfg,
        landmark106_cfg,
        landmark203_cfg,
        landmark478_cfg,
        appearance_extractor_cfg,
        motion_extractor_cfg,
    ):
        self.insightface_det = InsightFaceDet(**insightface_det_cfg)
        self.landmark106 = Landmark106(**landmark106_cfg)
        self.landmark203 = Landmark203(**landmark203_cfg)
        self.landmark478 = Landmark478(**landmark478_cfg)

        self.appearance_extractor = AppearanceExtractor(**appearance_extractor_cfg)
        self.motion_extractor = MotionExtractor(**motion_extractor_cfg)

    def close(self):
        self.landmark478.close()

    def _crop(self, img, last_lmk=None, **kwargs):
        # img_rgb -> det->landmark106->landmark203->crop

        if last_lmk is None:  # det for first frame or image
            det, _ = self.insightface_det(img)
            boxes = det[np.argsort(-(det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))]
            if len(boxes) == 0:
                return None
            lmk_for_track = self.landmark106(img, boxes[0])  # 106
        else:  # track for video frames
            lmk_for_track = last_lmk  # 203

        crop_dct = crop_image(
            img,
            lmk_for_track,
            dsize=self.landmark203.dsize,
            scale=1.5,
            vy_ratio=-0.1,
            pt_crop_flag=False,
        )
        lmk203 = self.landmark203(crop_dct["img_crop"], crop_dct["M_c2o"])

        ret_dct = crop_image(
            img,
            lmk203,
            dsize=512,
            scale=kwargs.get("crop_scale", 2.3),
            vx_ratio=kwargs.get("crop_vx_ratio", 0),
            vy_ratio=kwargs.get("crop_vy_ratio", -0.125),
            flag_do_rot=kwargs.get("crop_flag_do_rot", True),
            pt_crop_flag=False,
        )

        img_crop = ret_dct["img_crop"]
        M_c2o = ret_dct["M_c2o"]

        return img_crop, M_c2o, lmk203
    
    @staticmethod
    def _img_crop_to_bchw256(img_crop):
        rgb_256 = cv2.resize(img_crop, (256, 256), interpolation=cv2.INTER_AREA)
        rgb_256_bchw = (rgb_256.astype(np.float32) / 255.0)[None].transpose(0, 3, 1, 2)
        return rgb_256_bchw

    def _get_kp_info(self, img):
        # rgb_256_bchw_norm01
        kp_info = self.motion_extractor(img)
        return kp_info

    def _get_f3d(self, img):
        # rgb_256_bchw_norm01
        fs = self.appearance_extractor(img)
        return fs

    def _get_eye_info(self, img):
        # rgb uint8
        lmk478 = self.landmark478(img)  # [1, 478, 3]
        attr = EyeAttrUtilsByMP(lmk478)
        lr_open = attr.LR_open().reshape(-1, 2)   # [1, 2]
        lr_ball = attr.LR_ball_move().reshape(-1, 6)   # [1, 3, 2] -> [1, 6]
        return [lr_open, lr_ball]

    def __call__(self, img, last_lmk=None, **kwargs):
        """
        img: rgb, uint8
        last_lmk: last frame lmk203, for video tracking
        kwargs: optional crop cfg
            crop_scale: 2.3
            crop_vx_ratio: 0
            crop_vy_ratio: -0.125
            crop_flag_do_rot: True
        """
        img_crop, M_c2o, lmk203 = self._crop(img, last_lmk=last_lmk, **kwargs)

        eye_open, eye_ball = self._get_eye_info(img_crop)

        rgb_256_bchw = self._img_crop_to_bchw256(img_crop)
        kp_info = self._get_kp_info(rgb_256_bchw)
        fs = self._get_f3d(rgb_256_bchw)
        
        source_info = {
            "x_s_info": kp_info,
            "f_s": fs,
            "M_c2o": M_c2o,
            "eye_open": eye_open,   # [1, 2]
            "eye_ball": eye_ball,    # [1, 6]
            "lmk203": lmk203,  # for track
        }
        return source_info
