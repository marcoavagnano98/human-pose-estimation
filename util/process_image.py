import cv2
import numpy as np
from .skeletons import *
import matplotlib.pyplot as plt
import math
import os



class ImageProcessor():
  def exposure_correction(self, image):
    return cv2.convertScaleAbs(image, alpha=1.5, beta=10)
  
  def normalize_brightness(self, image):
    return cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  
  def channel_equalizer(self, image):
    equalized_channels = [cv2.equalizeHist(channel) for channel in cv2.split(image)]
    return cv2.merge(equalized_channels) 

def show_image(img):
   plt.ion()
   plt.imshow(img) 
   plt.waitforbuttonpress()
   
def add_patch(img,top, left, height,width):
    h, w,_ = img.shape
    assert height <= h and width <= w
    img[top:height, left:width, :] = 0 
    return img

def draw_landmarks(img, points, low_parts = 0, skeleton = "coco"):
  colors ={"blue": (255,0,0), "green": (0,255,0), "red":(0,0,255)}
  bones = get_skeleton(skeleton)
  for node in bones:
    p1, p2 = node
    if p1 >= low_parts and p2 >= low_parts:
      y1, x1, _ = points[node[0]]
      y2, x2, _ = points[node[1]]
      cv2.line(img, (int(x1), int(y1)),(int(x2), int(y2)),colors["blue"] , 4)

  for node in range(low_parts, len(points)):
      y1, x1, _ = points[node]
      cv2.putText(img, str(node), (int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors["red"], 2, cv2.LINE_AA)

  return img

def get_pixel_from_percent(dims, x, y):
  h, w = dims
  x_px = min(math.floor(x * w), w-1)
  y_px = min(math.floor(y * h), h-1)
  return x_px, y_px

def custom_cut(image, cut_type, keypoints):
  # neck type: we have to get a middle point between head and shoulder (0, min(rshoulder, left_shoulder))
  h, w, _ = image.shape
  if cut_type == "neck":
    y1, y2 = keypoints[0][0], min(keypoints[5][0], keypoints[6][0])
    y = np.mean([y1, y2])
  if cut_type == "shoulder":
     y = max(keypoints[5][0], keypoints[6][0])

  if cut_type == "chest" or cut_type == "above_hips":
    y1, y2 = max(keypoints[5][0], keypoints[6][0]), min(keypoints[11][0], keypoints[12][0])
    y_mean = np.mean([y1, y2])
    abs_dist = abs(y1 - y2)
    shift = abs_dist / 4
    if cut_type == "chest":
      y = y_mean - shift
    else:
      y = y_mean + shift
      
  if cut_type == "bottom_hips" or cut_type == "above_knee":
    y1, y2 = min(keypoints[11][0], keypoints[12][0]), max(keypoints[13][0], keypoints[14][0])
    y_mean = np.mean([y1, y2])
    abs_dist = abs(y1 - y2)
    shift = abs_dist / 4

    if cut_type == "bottom_hips":
      y = y_mean - shift
    else:
      y = y_mean + shift
     



  image = add_patch(image, 0, 0, int(y), w)
  return image
     
def black_to_gray():
  import cv2
  image = cv2.imread("dataset/youtube/dataset_nobg/youtube_0.jpg")#np.zeros((128,128,3), dtype=np.uint8)
  image1 = np.ones_like(image) * 127
  cv2.imshow("", image)
  cv2.waitKey(0)
  image = np.where(image == 0, image1, image)
  cv2.imshow("", image)
  cv2.waitKey(0)