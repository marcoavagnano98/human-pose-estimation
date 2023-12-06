import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import math
import cv2
import os

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display
dims = 192   
w,h = dims, dims
input_size = 1
#folder_path = "/home/marco/technogym/ava"



interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video.avi', fourcc, 30, (192, 192))

import time

def current_milli_time():
    return round(time.time() * 1000)

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    
    input_details = interpreter.get_input_details()
    
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores
def trasform_to_pix(x,y):
    x_px = min(math.floor(x * dims), dims-1) # 
    y_px = min(math.floor(y * dims), dims-1)
    return x_px, y_px

def draw_landmarks(img, points):
    lb_graph = [(12,11), (12, 14), (14, 16), (11, 13), (13, 15)]
    for node in lb_graph:
        y1, x1, conf1 = points[0][0][node[0]]
        y2, x2, conf2 = points[0][0][node[1]]
        y1,x1 = trasform_to_pix(y1, x1)
        y2,x2 = trasform_to_pix(y2, x2)
        cv2.line(img[0], (x1, y1),(x2, y2), (255,0,0), 2)
    return img[0]
     # 11, 13 15



def process_folder(folder_path):
    t = current_milli_time()
    elems = os.listdir(folder_path)
    elems = len([elem for elem in elems if not os.path.isdir(elem)])
    for i in range(elems):
        rp = os.path.join(folder_path,f"frame{i}.png")
        if os.path.exists(rp):
            img = cv2.imread(rp)
            img = cv2.resize(img,(dims,dims), cv2.INTER_AREA)
            img = np.expand_dims(img, axis=0)
            points = movenet(img)
            img = draw_landmarks(img, points)
            video.write(img)

    print(current_milli_time() - t)

def compute_angles(points):
    lb_graph = [(11, 13, 15), (12,11,13)] # , (12, 14,16),
    degs = []
    for node in lb_graph:
        # construct vectors 
        y1, x1, _ = points[0][0][node[0]]
        y2, x2, _ = points[0][0][node[1]]
        y3, x3, _ = points[0][0][node[2]]
        y1,x1 = trasform_to_pix(y1, x1)
        y2,x2 = trasform_to_pix(y2, x2)
        y3,x3 = trasform_to_pix(y3, x3)
        U = np.array([x1-x2, y1-y2])
        V =np.array([x3-x2, y3-y2])
        W = sum(U * V)
        Um =  math.sqrt(sum(U **2))
        Vm =  math.sqrt(sum(V **2))
        rad_angle = math.acos((W / (Um * Vm)))
        deg_angle = rad_angle * (180/math.pi)
        degs.append(deg_angle)
    return degs

def process_image(path = "../mediapipe/concat2.png"):
    # first path /home/marco/technogym/back/frame33.png
    # torso path /home/marco/Pictures/torso.png
    img = cv2.imread(path)

    #img2 = cv2.imread("/home/marco/Pictures/torso.png")
    #img2 = img2[100:, 102:]
    #torso_h, torso_w, _ = img2.shape
    #img2 = img2[:,: torso_w - 102]
   # img = cv2.vconcat([img2,img])
    #img = img[100:]
    #cv2.imshow("", cv2.vconcat([img2,img]))
    #cv2.waitKey(0)
    #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img,(192,192), cv2.INTER_AREA)
    img = np.expand_dims(img, axis=0)
    points = movenet(img)
    cv2.imwrite("ann2.png",draw_landmarks(img, points))
    
    rad, deg = compute_angles(points)
process_image()
#process_folder("/home/marco/technogym/left")
#cv2.destroyAllWindows()
#video.release()