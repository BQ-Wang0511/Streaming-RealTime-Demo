from .modules.retinaface import RetinaFace


class InsightFaceDet:
    def __init__(self, model_path, device="cuda", **_):
        self.model = RetinaFace(model_file=model_path, device=device)

    def __call__(self, img, **kwargs):
        return self.model.detect(img, **kwargs)
