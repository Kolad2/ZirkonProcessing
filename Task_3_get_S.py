import math
import os
import numpy as np
import cv2
from scipy.io import savemat

from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt
from StatisticEstimation import GetThetaLognorm
from scipy import stats as st

Path_dir_edges = "/media/kolad/HardDisk/Zirkon/ZirkonUpscaleBINEdgesPrep"
Path_dir_imgs = "/media/kolad/HardDisk/Zirkon/ZirkonUpscale"
Path_dir_segs = "/media/kolad/HardDisk/Zirkon/ZirkonUpscaleSegmentation"


FileNames = os.listdir(Path_dir_edges)
print(FileNames)

for FileName in FileNames:
       print(FileName)
       Path_img_edges = Path_dir_edges + "/" + FileName
       Path_img = Path_dir_imgs + "/" + FileName[0:9] + ".tif"
       Path_img_seg = Path_dir_segs+ "/" + FileName[0:9] + "_segs.tif"

       img_edges = cv2.imread(Path_img_edges)
       img = cv2.imread(Path_img)

       B, G, R = cv2.split(img_edges)

       G = 255 - G

       G[((R==255) & (B==0))] = 255
       _, area_marks = cv2.connectedComponents(255-R)
       area_marks = area_marks + 1
       area_marks[area_marks == 1] = 0
       area_marks[((R==255) & (B==0))] = 1
       area_marks = cv2.watershed(img_edges*0, area_marks)
       area_marks = area_marks - 1
       area_marks[area_marks == -2] = 0
       area_marks[area_marks == -1] = 0
       #
       unique, S = np.unique(area_marks, return_counts=True)
       save_dict = {'S': S[1:]}
       savemat("temp/" + FileName[0:9] + "_S.mat", save_dict)
       #
       rng = np.random.default_rng()
       MR = np.empty(area_marks.shape, np.uint8)
       MG = np.empty(area_marks.shape, np.uint8)
       MB = np.empty(area_marks.shape, np.uint8)
       for p in np.unique(area_marks):
              MR[area_marks == p] = rng.integers(0, 255)
              MG[area_marks == p] = rng.integers(0, 255)
              MB[area_marks == p] = rng.integers(0, 255)
       MR[area_marks == 0] = 0
       MG[area_marks == 0] = 0
       MB[area_marks == 0] = 0
       #
       img_M = cv2.merge((MR, MG, MB))
       cv2.imwrite(Path_img_seg , img_M)





