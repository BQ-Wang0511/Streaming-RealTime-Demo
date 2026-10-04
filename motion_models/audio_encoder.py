from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride,
        padding,
        residual: bool = False,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.conv_block(values)
        if self.residual:
            output = output + values
        return self.act(output)


class AudioFeatureEncoder(nn.Module):
    """Audio encoder used to obtain one 512-D feature per video frame."""

    def __init__(self):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1, residual=True),
            ConvBlock(32, 32, 3, 1, 1, residual=True),
            ConvBlock(32, 64, 3, (3, 1), 1),
            ConvBlock(64, 64, 3, 1, 1, residual=True),
            ConvBlock(64, 64, 3, 1, 1, residual=True),
            ConvBlock(64, 128, 3, 3, 1),
            ConvBlock(128, 128, 3, 1, 1, residual=True),
            ConvBlock(128, 128, 3, 1, 1, residual=True),
            ConvBlock(128, 256, 3, (3, 2), 1),
            ConvBlock(256, 256, 3, 1, 1, residual=True),
            ConvBlock(256, 512, 3, 1, 0),
            ConvBlock(512, 512, 1, 1, 0),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(mel).squeeze(2).squeeze(2)
