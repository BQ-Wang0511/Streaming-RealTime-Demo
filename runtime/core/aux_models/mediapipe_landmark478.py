from .modules.landmark478 import Landmark478 as Landmark478Model


class Landmark478:
    def __init__(self, task_path, **_):
        self.model = Landmark478Model(task_path=task_path)

    def __call__(self, image):
        result = self.model.detect_from_npimage(image.copy())
        return self.model.mplmk_to_nplmk(result)

    def close(self):
        self.model.close()
