import tensorflow as tf
import cv2
import pathlib
import sys
import numpy as np

sys.path.append("..")

from util.process_image import get_pixel_from_percent

parent_dir = pathlib.Path(__file__).parent.resolve()

class MoveNet():
    def __init__(self):
        self.dim = 192
        self.interpreter = tf.lite.Interpreter(model_path=f"{parent_dir}/model.tflite")
        self.interpreter.allocate_tensors()
    
    def preprocess(self, image):
        image = cv2.resize(image, (self.dim, self.dim), interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        tensor = tf.cast(image, dtype=tf.uint8)
        return tensor
    def normalize_results(self, keypoints, dims):
        res = []
        for kp in keypoints[0][0]:
            y,x,conf = kp
            x, y = get_pixel_from_percent(dims, x, y)
            res.append([y, x, conf])
        return np.array([res])



    def predict(self, image):
        orig_dims = image.shape[:2]
        tensor = self.preprocess(image)
        input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'], tensor.numpy())
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        return self.normalize_results(keypoints_with_scores, orig_dims)
    
def get_model():
    return MoveNet()