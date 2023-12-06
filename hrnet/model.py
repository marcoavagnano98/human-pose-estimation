import torch
from . import SimpleHRNet as SR
import pathlib
parent_f = pathlib.Path(__file__).parent.resolve()

def get_model():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return SR.SimpleHRNet(48, 17, f"{parent_f}/weights/pose_hrnet_w48_384x288.pth", multiperson=False, device=device)