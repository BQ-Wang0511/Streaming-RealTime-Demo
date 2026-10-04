import torch
from ..utils.load_model import load_model


class AppearanceExtractor:
    def __init__(self, model_path, device="cuda"):
        kwargs = {
            "module_name": "AppearanceFeatureExtractor",
        }
        self.model = load_model(model_path, device=device, **kwargs)
        self.device = device

    def __call__(self, image):
        """
        image: np.ndarray, shape (1, 3, 256, 256), float32, range [0, 1]
        """
        with torch.no_grad(), torch.autocast(
            device_type=self.device[:4], dtype=torch.float16, enabled=True
        ):
            pred = self.model(torch.from_numpy(image).to(self.device)).float().cpu().numpy()
        return pred
