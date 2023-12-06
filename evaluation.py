import math
import json
import numpy as np
from image_segmentation.deep_segmentation import DeepSegmentation
import cv2
import collections
import pathlib
from util.skeletons import *
from util.process_image import *

class HPEvaluator():
    def __init__(self,skeleton="coco"):
        # std for each keypoints in coco skeleton pose
        self.dir = pathlib.Path(__file__).parent.resolve()
        self.kp_std_coco = [0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072, 0.062,0.062, 0.107,0.107,0.087,0.087,0.089,0.089]
        self.kp_std_mp = [0.026, 0.025,0.025,0.025,0.025,0.025,0.025,
                          0.035,0.035,0.031,0.031,0.079,0.079,0.072,0.072,0.062,0.062,0.062,0.062,0.062,0.062,0.062,0.062, 0.107,0.107,0.087,0.087,0.089,0.089,0.092,0.092,0.092,0.092]
        
        self.load_skeleton_graph(skeleton)
        self.sk_name = skeleton
        self.segmentation = DeepSegmentation()


    def load_skeleton_graph(self, skeleton):
        sk = get_skeleton(skeleton)
        self.skeleton = {"name": skeleton, "len": 17, "edges": sk, "std":  self.kp_std_coco}

    
    def get_img_scale(self, image):
        mask = self.segmentation.get_mask(image)
        return (np.uint8(mask) >0).sum()


    def oks(self, gt_image, cut=0, pred_kp=None, true_kp=None): # object keypoints similarity
       # vis_factor = lambda conf: 0 if conf < 0.3 else 1
        vis = np.zeros(self.skeleton["len"])

        # we need to keep only coco keypoints to evaluate with this metrics because no std keypoints values are available for other skeletons
        true_kp, pred_kp = map_to_coco(self.skeleton["name"], true_kp, pred_kp)
        vis[cut:] = 1

        ## to apply ground truth images
        ksi = []
        scale = self.get_img_scale(gt_image)
        for limb, std_i in enumerate(self.skeleton["std"]):        
            # compute euclidean distance
            y1, x1, _ = pred_kp[limb]
            y2, x2, conf = true_kp[limb]
            eucl = math.dist((x1, y1), (x2, y2)) ** 2

            k = (std_i*2) ** 2 
            ksi.append(math.e ** (-eucl / (2 * scale *  k)))
        oks = 0
        vis_sum = vis.sum()
        
        if vis_sum == 0:
            return 0
        
        for ks, vf  in zip(ksi, vis):
            oks += (ks * vf) / vis_sum 
        return oks    
    
    def euclidean_dist(self, dims, pred_ks, true_ks, cut= 11):
        pred_ks, true_ks = pred_ks[cut:], true_ks[cut:]
        res = {}
        for idx, packed in enumerate(zip(pred_ks, true_ks)):
           pred, true = packed
           x1, y1, x2, y2 = (pred[0],pred[1], true[0], true[1])
           eucl = math.dist((x1, y1), (x2, y2))
           div_factor = math.sqrt((dims[0] -  dims[1]) **2)
           name = get_point_name("coco", idx+cut)
           res[name] = eucl / div_factor

        return {"mean": np.mean(list(res.values())), "scores": res}
           