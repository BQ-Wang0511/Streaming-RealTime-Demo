# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
from __future__ import division

import glob
import os.path as osp

import onnxruntime

from ..model_zoo import model_zoo
from .common import Face


DEFAULT_MP_NAME = 'buffalo_l'
__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, root='~/.insightface', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        model_paths = kwargs.pop("model_paths", None)
        if model_paths is None:
            self.model_dir = osp.join(osp.expanduser(root), 'models', name)
            if not osp.isdir(self.model_dir):
                raise FileNotFoundError(f"Face model directory not found: {self.model_dir}")
            onnx_files = sorted(glob.glob(osp.join(self.model_dir, '*.onnx')))
        else:
            self.model_dir = None
            onnx_files = [osp.abspath(path) for path in model_paths]
            missing = [path for path in onnx_files if not osp.isfile(path)]
            if missing:
                raise FileNotFoundError(f"Face model not found: {missing[0]}")
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                print('model ignore:', onnx_file, model.taskname)
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                self.models[model.taskname] = model
            else:
                print('duplicated model task type, ignore:', onnx_file, model.taskname)
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    def get(self, img, max_num=0):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        return ret

    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                for landmark_index in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if landmark_index == 0 or landmark_index == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[landmark_index][0], kps[landmark_index][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        return dimg
