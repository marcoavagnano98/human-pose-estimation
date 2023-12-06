import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pathlib
import numpy as np

class MpModel():
    def __init__(self):
        parent = pathlib.Path(__file__).parent.resolve()

        base_options = python.BaseOptions(model_asset_path=f'{parent}/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence = 0.0,
            output_segmentation_masks=True)

        self.detector = vision.PoseLandmarker.create_from_options(options)
    
    def predict(self, image):
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(rgb_frame)
        return self.normalize_results(image, detection_result.pose_landmarks[0])

    def normalize_results(self, image, pose_landmarks):
        results = []
        h, w, _ = image.shape
        for keypoint in pose_landmarks:
           x, y, conf = keypoint.x, keypoint.y, (keypoint.presence + keypoint.visibility) / 2
           x = (x* w) // 1
           y = (y * h) // 1
           results.append([y, x, conf])
        return np.array([results])



def get_model():
    return MpModel()

