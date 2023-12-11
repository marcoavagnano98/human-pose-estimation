from mediapipe import solutions
import numpy as np


kp_map_to_coco = {
    "coco": np.arange(17),
    "mediapipe": np.array([0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28]),
    "openpose": np.array([0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10])
}
coco_kp_names = ["Nose","Left eye","Right eye", "Left ear", "Right ear", "Left shoulder","Right shoulder", "Left elbow", "Right elbow", "Left hand", "Right hand", "Left hip", "Right hip", "Left knee","Right knee","Left ankle", "Right ankle"]


def get_skeleton(name):
    if name == "coco":
        return [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6]]
    
    if name == "mediapipe":
        return list(solutions.pose.POSE_CONNECTIONS)
    
    if name == "openpose":
        return [[0, 1], [0,14], [0,15], [15, 17], [14, 16], [1, 2], [1, 5], [1,8], [1,11], [5, 6], [5, 7], [2, 3], [2, 4], [8, 11], [11, 12], [12, 13], [8, 9], [9, 10], [2, 8], [5, 11]]
    
    
def get_point_name(name, idx):
    coco_kp = get_coco_keypoint(name, idx)
    return coco_kp_names[coco_kp]

# get coco index of given skeleton index
def get_coco_keypoint(skeleton, point):
    usable_kp = kp_map_to_coco[skeleton]

    assert point < len(usable_kp)

    return usable_kp[point]

# map keypoints from different skeletons to COCO skeleton
def map_to_coco(skeleton, true_kp, pred_kp):
    usable_kp = kp_map_to_coco[skeleton]

    assert len(usable_kp) <= len(true_kp)
    true_kp = true_kp[usable_kp]
    pred_kp = pred_kp[usable_kp]
    
    return true_kp, pred_kp