import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras import backend as K
from glob import glob
import cv2
import os
from tqdm import tqdm
import pathlib


class DeepSegmentation():
    def __init__(self):
        self.H = 512
        self.W = 512
        self.smooth = 1e-15
        tf.config.experimental.set_visible_devices([], 'GPU')
        parent = pathlib.Path(__file__).parent.resolve()
        model_p = f"{parent}/weights/model.h5"
        with CustomObjectScope({'iou': self.iou, 'dice_coef': self.dice_coef, 'dice_loss': self.dice_loss}):
            self.model = tf.keras.models.load_model(model_p)

    def iou(self, y_true, y_pred):
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()
            union = y_true.sum() + y_pred.sum() - intersection
            x = (intersection + 1e-15) / (union + 1e-15)
            x = x.astype(np.float32)
            return x
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)


    def dice_coef(self, y_true, y_pred):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred)
        return (2. * intersection + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)

    def dice_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef(y_true, y_pred)
    
    def get_mask(self, image):
        h, w, _ = image.shape
        x = cv2.resize(image, (self.W, self.H))
        x = x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        y = self.model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        return y

    def mask_generator(self, img_folder):
        data_x = glob(f"{img_folder}/*.png")
        for path in tqdm(data_x, total=len(data_x)):
            name = path.split("/")[-1].split(".")[0]
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            h, w, _ = image.shape
            x = cv2.resize(image, (self.W, self.H))
            x = x/255.0
            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=0)
            y = self.model.predict(x)[0]
            y = cv2.resize(y, (w, h))
            yield y
