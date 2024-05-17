import math
import os
import numpy as np
import cv2
from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt

Path_dir = "/media/kolad/HardDisk/Zirkon"
FileName1 = "Z-20eup8.png"
FileName2 = "Z-20eup8_edge_4.png"


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

#f_bins = 1*np.logspace(0,4,30,base=2)
f_bins = np.linspace(10, 10000,50)
f, _ = np.histogram(S, bins=f_bins, density=True)
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



# A = np.ones(area_marks.shape, np.uint8)*255
# A[((R==255) & (B==0))] = 0

# fig = plt.figure(figsize=(14, 9))
# axs = [fig.add_subplot(2, 2, 1),
#        fig.add_subplot(2, 2, 2),
#        fig.add_subplot(2, 2, 3),
#        fig.add_subplot(2, 2, 4)]
# axs[0].imshow(img1)
# axs[1].imshow(img2)
# axs[2].imshow(cv2.merge((G, G, G)))
# axs[3].imshow(img1)
# axs[3].imshow(img_M, alpha=0.5)

fig = plt.figure(figsize=(14, 9))
axs = [fig.add_subplot(1, 2, 1),
       fig.add_subplot(1, 2, 2)]
axs[0].imshow(img1)
axs[0].imshow(img_M, alpha=0.5)

print(len((f_bins[1:]+f_bins[0:-1])/2),len(f))

fx = (f_bins[1:]+f_bins[0:-1])/2
axs[1].plot(fx,f)
axs[1].set_xscale('log')
axs[1].set_yscale('log')

plt.show()