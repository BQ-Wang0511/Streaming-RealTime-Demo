import torch
from ..utils.load_model import load_model


class WarpNetwork:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "WarpingNetwork",
        }
        self.model = load_model(model_path, device=device, **kwargs)
        self.device = device
        self._cached_feature_id = None
        self._cached_feature = None

    def __call__(self, feature_3d, kp_source, kp_driving):
        """
        feature_3d: np.ndarray, shape (1, 32, 16, 64, 64)
        kp_source | kp_driving: np.ndarray, shape (1, 21, 3)
        """
        feature_id = id(feature_3d)
        if feature_id != self._cached_feature_id:
            self._cached_feature = torch.from_numpy(feature_3d).to(self.device)
            self._cached_feature_id = feature_id
        with torch.inference_mode(), torch.autocast(
            device_type=self.device[:4], dtype=torch.float16, enabled=True
        ):
            pred = self.model(
                self._cached_feature,
                torch.from_numpy(kp_source).to(self.device),
                torch.from_numpy(kp_driving).to(self.device),
            )
        return pred
