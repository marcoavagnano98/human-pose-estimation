import pathlib
import numpy as np
import sys
sys.path.append('.')
parent_dir = pathlib.Path(__file__).parent.resolve()

from .src.body import Body
class OpenPosePyTorch():


    def __init__(self):
        self.model_path = f"{parent_dir}/model/body_pose_model.pth"
        self.body_estimation = Body(self.model_path)
    def normalize_parts(self, candidates):
        cand = candidates[:,:3]
        points = [[part[1], part[0], part[2]] for part in cand]
        return np.array([points])
    
    def predict(self, image):
        cand = self.body_estimation(image)
        cand = self.normalize_parts(cand)
        return cand



def get_model():
    return OpenPosePyTorch()


