import cv2
import numpy as np
import pathlib 
from evaluation import HPEvaluator
from util.process_image import *
from util.skeletons import *
import argparse
import random
import os
from util.json_writer import *
from pathlib import Path

parser = argparse.ArgumentParser(prog="Human Pose Estimation")
parser.add_argument("-test",  action='store_true')
parser.add_argument("-folder")
parser.add_argument("-estimators")
parser.add_argument("-read")
parser.add_argument("-write", action='store_true')
parser.add_argument("-show", action='store_true')
parser.add_argument("-cut")
args = parser.parse_args()

DEFAULT_PATH = "dataset/youtube/real"


def process_args(arg):
    import re
    arg = re.sub(r'\s', '', arg)
    arg = arg.split(',')
    return arg

class HumanPoseEstimation():
    def __init__(self, estimator, dataset_size, write = False, read = False):
        self.dir = pathlib.Path(__file__).parent.resolve() 
        self.write = write
        self.read = read
        if write:
            self.writer = Writer()
        
        if read:
            self.reader = Reader()
            print(f"Loading {read} annotations")
            self.load_reader(read)

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

    def load_reader(self, folder_p):
        for fname in os.listdir(folder_p):
            p = os.path.join(folder_p, fname)
            self.reader.load_file(p)

    def get_keypoints(self, image):
        return self.model.predict(image)

    def show_diff(self, gt_image, predicted, kp_true, kp_predicted):
        img1 = draw_landmarks(predicted, kp_predicted)
        img2 = draw_landmarks(gt_image, kp_true)
        show_image(img1)
        show_image(img2)

    def handle_write(self, image_id, keypoints, limit = 1000):
        if self.image_counter == (self.dataset_size - 1):
            self.writer.buffer_data(image_id, keypoints)
            self.writer.close()
        else:
            if self.image_counter == 0:
                if self.writer.is_open():
                    self.writer.close()
                fname = f"{self.image_counter}-{(self.image_counter + limit) - 1}.json"
                self.writer.begin(fname)
            
            self.writer.buffer_data(image_id, keypoints)
    

    def get_keypoints_from_file(self, image_id):
        return self.reader.get_keypoints(image_id)

    def read_keypoints(self, image_id):
        return self.reader.get_keypoints(image_id)

    def test(self, cut, image_path, show = False):
        """
        id = get_image_id(image_path)
        keypoints = self.read_keypoints(id)
        print(keypoints)
        return 0
        """
        image_id = get_image_id(image_path)
        gt_image = cv2.imread(image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        _, gtw, _ = gt_image.shape
        if not self.read:
            # Compute the keypoints only if a file containing them is not present
            kp_true = self.get_keypoints(gt_image)[0]
        else:
            kp_true = np.array(self.get_keypoints_from_file(image_id))

        # cut the image on given kpoint y
     #   predicted, kp_start = custom_cut(gt_image.copy(), cut, kp_true)
     #   kp_predicted = self.get_keypoints(predicted)[0]
        """
        if show:
            # map skeleton of current HPE estimator to coco skeleton
            true_to_show, preds_to_show = map_to_coco(self.skeleton, kp_true.copy(), kp_predicted.copy())
            self.show_diff(gt_image, predicted, true_to_show, preds_to_show)
        """
        if self.write:
            self.handle_write(image_id, kp_true.tolist())
        self.image_counter += 1

        # EVALUATION 
        """
        oks = self.evaluator.oks(gt_image, kp_start, kp_predicted, kp_true)
        euclidean_dist = self.evaluator.euclidean_dist(gt_image.shape[:2], kp_predicted, kp_true)
        print(oks)
        """
    


if __name__ == "__main__":
    if args.test:
        folder_p = args.folder if args.folder else DEFAULT_PATH
        estimators = ["hrnet", "openpose", "movenet", "mediapipe"] if args.estimators == "all" else process_args(args.estimators)
        if args.cut:
            cut = args.cut    
        for estimator  in estimators:
            files = list(os.listdir(folder_p))
            hpe = HumanPoseEstimation(estimator=estimator, dataset_size=len(files), write = args.write, read = args.read)
            files.sort()

            for img in files:
                print(img)
                image_path = os.path.join(folder_p,img)
                print(hpe.test(cut=args.cut, image_path=image_path, show=args.show))