from .modules.landmark106 import Landmark106 as Landmark106Model


class Landmark106:
    def __init__(self, model_path, device="cuda", **_):
        self.model = Landmark106Model(model_file=model_path, device=device)

    def __call__(self, img, bbox):
        return self.model.get(img, bbox)
