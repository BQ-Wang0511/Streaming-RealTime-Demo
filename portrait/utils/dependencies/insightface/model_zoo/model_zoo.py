# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
import os
import os.path as osp
import glob
import onnxruntime
from .retinaface import RetinaFace
from .landmark import Landmark

__all__ = ['get_model']


class PickableInferenceSession(onnxruntime.InferenceSession):
    # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model_path = model_path

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        model_path = values['model_path']
        self.__init__(model_path)

class ModelRouter:
    def __init__(self, onnx_file):
        self.onnx_file = onnx_file

    def get_model(self, **kwargs):
        session = PickableInferenceSession(self.onnx_file, **kwargs)
        inputs = session.get_inputs()
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        outputs = session.get_outputs()

        if len(outputs)>=5:
            return RetinaFace(model_file=self.onnx_file, session=session)
        elif input_shape[2]==192 and input_shape[3]==192:
            return Landmark(model_file=self.onnx_file, session=session)
        else:
            return None

def find_onnx_file(dir_path):
    if not os.path.exists(dir_path):
        return None
    paths = glob.glob("%s/*.onnx" % dir_path)
    if len(paths) == 0:
        return None
    paths = sorted(paths)
    return paths[-1]

def get_default_providers():
    return ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']

def get_default_provider_options():
    return None

def get_model(name, **kwargs):
    root = kwargs.get('root', '~/.insightface')
    root = os.path.expanduser(root)
    model_root = osp.join(root, 'models')
    if not name.endswith('.onnx'):
        model_dir = os.path.join(model_root, name)
        model_file = find_onnx_file(model_dir)
        if model_file is None:
            return None
    else:
        model_file = name
    if not osp.isfile(model_file):
        raise FileNotFoundError(model_file)
    router = ModelRouter(model_file)
    providers = kwargs.get('providers', get_default_providers())
    provider_options = kwargs.get('provider_options', get_default_provider_options())
    model = router.get_model(providers=providers, provider_options=provider_options)
    return model
