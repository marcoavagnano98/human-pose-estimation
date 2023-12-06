import cv2
import numpy as np
import pathlib 
from evaluation import HPEvaluator
from util.process_image import *
from util.skeletons import *
import argparse
import random
import os
from util.json_writer import Writer
from pathlib import Path

parser = argparse.ArgumentParser(prog="Human Pose Estimation")
parser.add_argument("-test",  action='store_true')
parser.add_argument("-folder")
parser.add_argument("-estimators")
parser.add_argument("-cut")
args = parser.parse_args()

DEFAULT_PATH = "dataset/youtube/real"


def process_args(arg):
    import re
    arg = re.sub(r'\s', '', arg)
    arg = arg.split(',')
    return arg

class HumanPoseEstimation():
    def __init__(self, estimator, dataset_size):
        self.dir = pathlib.Path(__file__).parent.resolve() 
        self.writer = Writer()
        self.image_counter = 0
        self.dataset_size = dataset_size
        self.estimator = estimator
        self.end_reached = False
        self.skeleton = "coco"


        if estimator == "openpose":
            self.evaluator = HPEvaluator(skeleton=estimator)
            self.skeleton = estimator
            from openpose.model import get_model
            
        if estimator == "hrnet":
            self.evaluator = HPEvaluator()
            from hrnet.model import get_model

        if estimator == "mediapipe":
            self.skeleton = estimator
            self.evaluator = HPEvaluator(skeleton=estimator)
            from mp.model import get_model

        if estimator == "movenet":
            from movenet.model import get_model
           # self.evaluator = HPEvaluator()

        self.model = get_model()   

    def get_keypoints(self, image):
        return self.model.predict(image)

    def show_diff(self, gt_image, predicted, kp_true, kp_predicted):
        img1 = draw_landmarks(predicted, kp_predicted)
        img2 = draw_landmarks(gt_image, kp_true)
        show_image(img1)
        show_image(img2)

    def handle_write(self, image_id, keypoints, limit = 1000):
        image_id = os.path.basename(image_id)[:-4]
        if self.image_counter == (self.dataset_size - 1):
            self.writer.buffer_data(image_id, keypoints)
            self.writer.close()
        else:
            if self.image_counter % limit == 0:
                if self.writer.is_open():
                    self.writer.close()
                fname = f"{self.image_counter}-{(self.image_counter + limit) - 1}.json"
                self.writer.begin(fname)
            
            self.writer.buffer_data(image_id, keypoints)
        
     
           
    
        

    def test(self, cut, image_path, show = False):
        gt_image = cv2.imread(image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        _, gtw, _ = gt_image.shape
        #assert gt_image.shape == predicted.shape
        kp_true = self.get_keypoints(gt_image)[0]
        """
        if cut == -1:
            y = 0
            cut = 0
        else:
            y = kp_true[get_coco_keypoint(self.skeleton, cut)][0]
        """        
        # cut the image on given kpoint y
        predicted = custom_cut(gt_image.copy(), cut, kp_true)#add_patch(gt_image.copy(), 0,0, int(y), gtw)
        kp_predicted = self.get_keypoints(predicted)[0]
        if show:
            true_to_show, preds_to_show = map_to_coco(self.skeleton, kp_true.copy(), kp_predicted.copy())
            self.show_diff(gt_image, predicted, true_to_show, preds_to_show)
        self.handle_write(image_path, kp_true.tolist())
        self.image_counter += 1
        return self.image_counter #{"eucl": self.evaluator.euclidean_dist(gt_image.shape[:2], kp_predicted, kp_true)}#"oks": self.evaluator.oks(gt_image, cut, kp_predicted, kp_true)}
    


if __name__ == "__main__":
    if args.test:
        folder_p = args.folder if args.folder else DEFAULT_PATH
        estimators = ["hrnet", "openpose", "movenet", "mediapipe"] if args.estimators == "all" else process_args(args.estimators)
        for estimator  in estimators:
            files = list(os.listdir(folder_p))
            hpe = HumanPoseEstimation(estimator=estimator, dataset_size=len(files))
            files.sort()
            for img in files:
                print(img)
                image_path = os.path.join(folder_p,img)
                #print(hpe.test(cut="bottom_hips", image_path=image_path, show=False))    
            