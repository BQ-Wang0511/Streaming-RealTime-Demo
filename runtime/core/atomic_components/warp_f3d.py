from ..models.warp_network import WarpNetwork


"""
# __init__
warp_network_cfg = {
    "model_path": "",
    "device": "cuda",
}
"""

class WarpF3D:
    def __init__(
        self,
        warp_network_cfg,
    ):
        self.warp_net = WarpNetwork(**warp_network_cfg)

    def __call__(self, f_s, x_s, x_d):
        out = self.warp_net(f_s, x_s, x_d)
        return out

    
