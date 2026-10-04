import torch
from ..utils.load_model import load_model


class StitchNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "StitchingNetwork",
        }
        self.model = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, kp_source, kp_driving):
        with torch.no_grad():
            return self.model(
                torch.from_numpy(kp_source).to(self.device),
                torch.from_numpy(kp_driving).to(self.device),
            ).cpu().numpy()
