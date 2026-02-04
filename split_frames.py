# -------------------------------------------------
# SAMA, AAAI 2024
# Training code for VQA. 
# This code is modified from FAST-VQA [ECCV, 2022]
# -------------------------------------------------
import torch
import random
import os.path as osp
import fastvqa.models as models
import fastvqa.datasets as datasets
import os
import argparse
import sys

from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np

import timeit
import math

import yaml

from functools import reduce
from thop import profile
import warnings
import cv2

warnings.filterwarnings("ignore")

video_path = '/media/user1/Data 2/HaThu2025/VQA-datasets/InTheWild/LIVE-VQC/Video/B009.mp4'
save_folder = './split_frames/B009_frames'
os.makedirs(save_folder, exist_ok=True)
vidcap = cv2.VideoCapture(video_path)
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite(osp.join(save_folder, "frame_%04d.jpg" % count), image)     # save frame as JPG file
    success, image = vidcap.read()
    
    count += 1
if count > 0:
    print(f'Split {count} frames successfully and saved to {save_folder}')
else:
    print('No frames were split.')