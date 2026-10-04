from .modules.landmark203 import Landmark203 as Landmark203Model


class Landmark203:
    def __init__(self, model_path, device="cuda", **_):
        self.model = Landmark203Model(model_file=model_path, device=device)
        self.dsize = self.model.dsize

    def __call__(self, img_crop_rgb, M_c2o=None):
        return self.model.run(img_crop_rgb, M_c2o)
