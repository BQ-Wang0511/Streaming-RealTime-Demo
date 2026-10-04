import onnxruntime
import numpy as np


def _transform_pts(pts, M):
    """ conduct similarity or affine transformation to the pts
    pts: Nx2 ndarray
    M: 2x3 matrix or 3x3 matrix
    return: Nx2
    """
    return pts @ M[:2, :2].T + M[:2, 2]

    
class Landmark203:
    def __init__(self, model_file, device="cuda"):
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_file, providers=providers)

        self.dsize = 224

    def _run(self, inp):
        out = self.session.run(None, {'input': inp})
        return out
    
    def run(self, img_crop_rgb, M_c2o=None):
        # img_crop_rgb: 224x224

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)

        out_lst = self._run(inp)
        out_pts = out_lst[2]

        # 2d landmarks 203 points
        lmk = out_pts[0].reshape(-1, 2) * self.dsize  # scale to 0-224
        if M_c2o is not None:
            lmk = _transform_pts(lmk, M=M_c2o)

        return lmk
    
