import cv2
import numpy as np
from ..utils.blend import blend_images_cy
from ..utils.get_mask import get_mask


class PutBack:
    def __init__(
        self,
        mask_template_path=None,
    ):
        if mask_template_path is None:
            mask = get_mask(512, 512, 0.9, 0.9)
            mask = np.concatenate([mask] * 3, 2)
        else:
            mask = cv2.imread(mask_template_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        self.mask_ori_float = np.ascontiguousarray(mask)[:,:,0]
        self.result_buffer = None
        self.frame_warped_buffer = None
        self.mask_warped = None
        self.mask_cache_key = None

    def __call__(self, frame_rgb, render_image, M_c2o):
        h, w = frame_rgb.shape[:2]
        mask_cache_key = (id(M_c2o), h, w)
        if self.mask_cache_key != mask_cache_key:
            self.mask_warped = cv2.warpAffine(
                self.mask_ori_float,
                M_c2o[:2, :],
                dsize=(w, h),
                flags=cv2.INTER_LINEAR,
            ).clip(0, 1)
            self.mask_cache_key = mask_cache_key
        if (
            self.frame_warped_buffer is None
            or self.frame_warped_buffer.shape != (h, w, 3)
            or self.frame_warped_buffer.dtype != render_image.dtype
        ):
            self.frame_warped_buffer = np.empty(
                (h, w, 3),
                dtype=render_image.dtype,
            )
        cv2.warpAffine(
            render_image,
            M_c2o[:2, :],
            dsize=(w, h),
            dst=self.frame_warped_buffer,
            flags=cv2.INTER_LINEAR,
        )
        if self.result_buffer is None or self.result_buffer.shape != (h, w, 3):
            self.result_buffer = np.empty((h, w, 3), dtype=np.uint8)

        blend_images_cy(
            self.mask_warped,
            self.frame_warped_buffer,
            frame_rgb,
            self.result_buffer,
        )

        return self.result_buffer
