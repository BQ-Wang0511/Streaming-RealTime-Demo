"""Canonical keypoint, head-pose, and expression extractor."""

from torch import nn

from .convnextv2 import convnextv2_tiny

model_dict = {
    'convnextv2_tiny': convnextv2_tiny,
}


class MotionExtractor(nn.Module):
    def __init__(self, **kwargs):
        super(MotionExtractor, self).__init__()

        backbone = kwargs.get('backbone', 'convnextv2_tiny')
        self.detector = model_dict.get(backbone)(**kwargs)

    def forward(self, x):
        out = self.detector(x)
        return out
