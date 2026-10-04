# coding: utf-8

"""
Stitching module(S) and two retargeting modules(R) defined in the paper.

- The stitching module pastes the animated portrait back into the original image space without pixel misalignment, such as in
the stitching region.

- The eyes retargeting module is designed to address the issue of incomplete eye closure during cross-id reenactment, especially
when a person with small eyes drives a person with larger eyes.

- The lip retargeting module is designed similarly to the eye retargeting module, and can also normalize the input by ensuring that
the lips are in a closed state, which facilitates better animation driving.
"""
import torch
from torch import nn


def remove_ddp_dumplicate_key(state_dict):
    from collections import OrderedDict
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


class StitchingNetwork(nn.Module):
    def __init__(self, input_size=126, hidden_sizes=[128, 128, 64], output_size=65):
        super(StitchingNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.mlp = nn.Sequential(*layers)

    def _forward(self, x):
        return self.mlp(x)
    
    def load_model(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        self.eval()
        return self

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """
        bs, num_kp = kp_source.shape[:2]
        kp_driving_new = kp_driving.clone()
        delta = self._forward(torch.cat([kp_source.view(bs, -1), kp_driving_new.view(bs, -1)], dim=1))
        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2
        kp_driving_new += delta_exp
        kp_driving_new[..., :2] += delta_tx_ty
        return kp_driving_new
    
    def forward(self, kp_source, kp_driving):
        out = self.stitching(kp_source, kp_driving)
        return out