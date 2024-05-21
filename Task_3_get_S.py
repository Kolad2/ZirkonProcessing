import math
import os
import numpy as np
import cv2
from scipy.io import savemat

from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt
from StatisticEstimation import GetThetaLognorm
from scipy import stats as st

Path_dir = "/media/kolad/HardDisk/Zirkon"
FileName1 = "Z-20eup8.png"
FileName2 = "Z-20eup8_edge_0.png"


Path_img1 = Path_dir + "/" + FileName1
Path_img2 = Path_dir + "/" + FileName2
img1 = cv2.imread(Path_img1)
img2 = cv2.imread(Path_img2)



B, G, R = cv2.split(img2)

G = 255 - G

G[((R==255) & (B==0))] = 255
_, area_marks = cv2.connectedComponents(G)
print(np.unique(G), np.max(G))
area_marks = cv2.watershed(img1, area_marks)
#
unique, S = np.unique(area_marks, return_counts=True)

save_dict = {'S': S}
savemat("temp/" + "S.mat", save_dict)


#
rng = np.random.default_rng()
MR = np.empty(area_marks.shape, np.uint8)
MG = np.empty(area_marks.shape, np.uint8)
MB = np.empty(area_marks.shape, np.uint8)
for p in np.unique(area_marks):
       MR[area_marks == p] = rng.integers(0,255)
       MG[area_marks == p] = rng.integers(0, 255)
       MB[area_marks == p] = rng.integers(0, 255)
img_M = cv2.merge((MR, MG, MB))


fig = plt.figure(figsize=(14, 9))
axs = [fig.add_subplot(1, 1, 1)]
axs[0].imshow(img1)
axs[0].imshow(img_M, alpha=0.5)
plt.show()