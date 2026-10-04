import numpy as np
import torch
from ..utils.load_model import load_model


class Decoder:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "SPADEDecoder",
        }
        self.model = load_model(model_path, device=device, **kwargs)
        self.device = device
        
    def __call__(self, feature):
        tensor = (
            feature
            if torch.is_tensor(feature)
            else torch.from_numpy(feature).to(self.device)
        )
        with torch.inference_mode(), torch.autocast(
            device_type=self.device[:4], dtype=torch.float16, enabled=True
        ):
            pred = self.model(tensor).float().cpu().numpy()
        
        pred = np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255    # [h, w, c]
        
        return pred
