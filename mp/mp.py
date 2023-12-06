from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os

import time

def current_milli_time():
    return round(time.time() * 1000)

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



folder_path = "../back/"
annotated_path = f"{folder_path}annotated/"
outdir, filename = "images_with_landmarks", "annotated.avi"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video.avi', fourcc, 30, (424, 240))
sup_part = "torso.jpg"
torso = cv2.imread(sup_part)[150:250,:,:]
torso = torso[:,:torso.shape[1]-46,:]
original_image = "../left/frame5.png"
original_image = cv2.imread(original_image)
original_image = cv2.vconcat([torso, original_image])
#cv2.imwrite("concat2.png",original_image)

#output = cv2.VideoWriter(os.path.join(outdir, filename),cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,(frame_w, frame_h))

f_n = 0

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

start_t = current_milli_time()
# STEP 2: Create an PoseLandmarker object.
#for img_path in os.listdir(folder_path):
for i in range(1):
  img_path = os.path.join(folder_path,f"frame{i}.png")
       # image = cv2.imread(os.path.join(fp, f"{source}{i}.png"))  
  if os.path.isdir(img_path):
     continue
  #rel_path = os.path.join(folder_path, img_path)
  base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)

  # STEP 3: Load the input image.
  image = mp.Image.create_from_file("concat2.png")
  
  # STEP 4: Detect pose landmarks from the input image.
  
  detection_result = detector.detect(image)

  # STEP 5: Process the detection result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
  #cv2.imwrite(f"{annotated_path}-{f_n}.png", annotated_image)
  annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  cv2.imwrite("ann2.png",annotated_image)
  #video.write(annotated_image)
  print(f_n)
  f_n += 1
  #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
print(current_milli_time() - start_t)
cv2.destroyAllWindows()
video.release()