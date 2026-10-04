# coding: utf-8

"""
Warping field estimator(W) defined in the paper, which generates a warping field using the implicit
keypoint representations x_s and x_d, and employs this flow field to warp the source feature volume f_s.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .util import SameBlock2d
from .dense_motion import DenseMotionNetwork


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp=21,
        block_expansion=64,
        max_features=512,
        num_down_blocks=2,
        reshape_channel=32,
        estimate_occlusion_map=True,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        dense_motion_params = {
            "block_expansion": 32,
            "max_features": 1024,
            "num_blocks": 5,
            "reshape_depth": 16,
            "compress": 4,
        }

        self.dense_motion_network = DenseMotionNetwork(
            num_kp=num_kp,
            feature_channel=reshape_channel,
            estimate_occlusion_map=estimate_occlusion_map,
            **dense_motion_params
        )

        self.third = SameBlock2d(max_features, block_expansion * (2 ** num_down_blocks), kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks), out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map

    def deform_input(self, inp, deformation):
        return F.grid_sample(inp, deformation, align_corners=False)

    def forward(self, feature_3d, kp_source, kp_driving):
        # Feature warper, Transforming feature representation according to deformation and occlusion
        dense_motion = self.dense_motion_network(
            feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
        )
        if 'occlusion_map' in dense_motion:
            occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
        else:
            occlusion_map = None

        deformation = dense_motion['deformation']  # Bx16x64x64x3
        out = self.deform_input(feature_3d, deformation)  # Bx32x16x64x64

        bs, c, d, h, w = out.shape  # Bx32x16x64x64
        out = out.view(bs, c * d, h, w)  # -> Bx512x64x64
        out = self.third(out)  # -> Bx256x64x64
        out = self.fourth(out)  # -> Bx256x64x64

        if self.flag_use_occlusion_map and (occlusion_map is not None):
            out = out * occlusion_map

        return out
    
    def load_model(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        self.eval()
        return self
