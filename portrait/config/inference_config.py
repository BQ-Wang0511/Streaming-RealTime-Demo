# coding: utf-8

"""Human portrait model configuration used for motion-template extraction."""

import cv2
from dataclasses import dataclass, field
from numpy import ndarray
from typing import Tuple

from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)
class InferenceConfig(PrintableConfig):
    models_config: str = make_abs_path("./models.yaml")
    checkpoint_F: str = make_abs_path(
        "../../checkpoints/runtime/models/appearance_extractor.pth"
    )
    checkpoint_M: str = make_abs_path(
        "../../checkpoints/runtime/models/motion_extractor.pth"
    )
    checkpoint_G: str = make_abs_path(
        "../../checkpoints/runtime/models/decoder.pth"
    )
    checkpoint_W: str = make_abs_path(
        "../../checkpoints/runtime/models/warp_network.pth"
    )
    checkpoint_S: str = make_abs_path(
        "../../checkpoints/runtime/models/stitch_network.pth"
    )
    flag_use_half_precision: bool = True
    device_id: int = 0
    flag_force_cpu: bool = False
    flag_do_torch_compile: bool = False
    source_max_dim: int = 1280
    source_division: int = 2
    input_shape: Tuple[int, int] = (256, 256)
    mask_crop_path: str = make_abs_path("../utils/resources/mask_template.png")
    mask_crop: ndarray = field(
        default_factory=lambda: cv2.imread(
            make_abs_path("../utils/resources/mask_template.png"), cv2.IMREAD_COLOR
        )
    )
