import math
import os
import numpy as np
import cv2
from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt


def step_skeleton(edges_w):
	edges_0 = np.zeros(edges_w.shape, np.uint8)
	kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
	for i in range(60, 255, 1):
		print(i)
		ret, result_bin = cv2.threshold(edges_w, i, 255, cv2.THRESH_BINARY)
		result_bin = cv2.erode(result_bin, kernel, iterations=1)
		result_erod = cv2.erode(result_bin, kernel, iterations=1)
		result_erod = cv2.subtract(result_bin, result_erod)
		result_bin = cv2.add(result_bin, edges_0)
		edges = cv2.ximgproc.thinning(result_bin)
		edges = cv2.subtract(edges, 255 - result_erod)
		edges_0 = cv2.add(edges_0, edges)
	return 255 - edges_0


dx = 1000
dy = 1000
ddx = 256
ddy = 256
Path_dir = "/media/kolad/HardDisk/Zirkon"

FileNames = os.listdir(Path_dir)
print(FileNames)

for FileName in FileNames:
	FileName = "Z-20e.png"
	# print(FileName)
	# Path_dir = Path0 + "/" + FileName + "/"
	Path_img = Path_dir + "/" + FileName
	Path_img_edges = Path_dir + "/" + FileName[:-3] + ".tiff"
	img = cv2.imread(Path_img)
	model = modelgpu()
	result_rsf = model.get_model_edges(img)
	result_rsf = np.uint8((result_rsf / result_rsf.max()) * 255)
	#ret, result_rsfB = cv2.threshold(result_rsf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	result_rsfB = step_skeleton(result_rsf)
	#exit()
	fig = plt.figure(figsize=(14, 9))
	fig.suptitle(FileName, fontsize=16)
	axs = [fig.add_subplot(1, 2, 1),
	       fig.add_subplot(1, 2, 2)]
	axs[0].imshow(img)
	axs[1].imshow(img)
	alphas = np.ones(result_rsfB.shape)
	kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
	result_rsfB = cv2.erode(result_rsfB, kernel, iterations=1)
	alphas[result_rsfB == 255] = 0
	alphas[result_rsfB == 0] = 1
	axs[1].imshow(result_rsfB, alpha=alphas)
	plt.show()
	#
	#img_rsf = cv2.merge((result_rsf, result_rsf, result_rsf))
	#cv2.imwrite(Path_img_edges, img_rsf)
	exit()
