# coding: utf-8

"""
Motion extractor(M), which directly predicts the canonical keypoints, head pose and expression deformation of the input image
"""

from torch import nn
import torch

from .convnextv2 import convnextv2_tiny


class MotionExtractor(nn.Module):
    def __init__(self, num_kp=21, backbone="convnextv2_tiny"):
        super(MotionExtractor, self).__init__()
        self.detector = convnextv2_tiny(num_kp=num_kp, backbone=backbone)

    def forward(self, x):
        out = self.detector(x)
        return out  # pitch, yaw, roll, t, exp, scale, kp
    
    def load_model(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
        self.eval()
        return self
