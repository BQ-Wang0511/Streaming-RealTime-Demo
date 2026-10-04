import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


class Landmark478:
    def __init__(self, task_path):
        base_options = BaseOptions(model_asset_path=task_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        self.detector = detector

    def close(self):
        detector = self.detector
        self.detector = None
        if detector is not None:
            detector.close()

    def detect_from_imp(self, imp):
        if self.detector is None:
            raise RuntimeError("Landmark detector is closed")
        image = mp.Image.create_from_file(imp)
        detection_result = self.detector.detect(image)
        return detection_result

    def detect_from_npimage(self, img):
        if self.detector is None:
            raise RuntimeError("Landmark detector is closed")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = self.detector.detect(image)
        return detection_result

    @staticmethod
    def mplmk_to_nplmk(results):
        face_landmarks_list = results.face_landmarks
        np_lms = []
        for face_lms in face_landmarks_list:
            lms = [[lm.x, lm.y, lm.z] for lm in face_lms]
            np_lms.append(lms)
        return np.array(np_lms).astype(np.float32)
