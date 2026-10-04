from ..models.decoder import Decoder


"""
# __init__
decoder_cfg = {
    "model_path": "",
    "device": "cuda",
}
"""

class DecodeF3D:
    def __init__(
        self,
        decoder_cfg,
    ):
        self.decoder = Decoder(**decoder_cfg)

    def __call__(self, f_s):
        out = self.decoder(f_s)
        return out

    
