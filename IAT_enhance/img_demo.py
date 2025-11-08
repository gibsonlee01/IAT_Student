import os 
import torch
import cv2
import argparse
import warnings
import torchvision
import numpy as np
from utils import PSNR, validation, LossNetwork
from model.IAT_main import IAT
from torchvision.transforms import Normalize
import matplotlib.pyplot as plt
from PIL import Image
from model.IAT_student import IAT_Student_BN

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='demo_imgs/low_demo.png')
parser.add_argument('--normalize', type=bool, default=True) #you should normalize!
config = parser.parse_args()

# Weights path
student_pretrain = r'/content/drive/MyDrive/IAT_test/IAT_enhance/ckpts/IAT_student/student_best.pth'

normalize_process = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Load Pre-train Weights

checkpoint = torch.load(student_pretrain, map_location='cuda', weights_only = False)
model = IAT_Student_BN().cuda()
model.load_state_dict(checkpoint['state_dict'])  
model.eval()


## Load Image
img = Image.open(config.file_name)
img = (np.asarray(img)/ 255.0)
if img.shape[2] == 4:
    img = img[:,:,:3]
input = torch.from_numpy(img).float().cuda()
input = input.permute(2,0,1).unsqueeze(0)
if config.normalize:
    input = normalize_process(input)

## Forward Network
_, _ ,enhanced_img = model(input)

torchvision.utils.save_image(enhanced_img, 'result.png')