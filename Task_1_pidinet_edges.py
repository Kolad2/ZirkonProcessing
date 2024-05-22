import math
import os
import numpy as np
import cv2
from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt


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
	#ret, result_rsfB = cv2.threshold(result_rcf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	fig = plt.figure(figsize=(14, 9))
	fig.suptitle(FileName, fontsize=16)
	axs = [fig.add_subplot(1, 2, 1),
	       fig.add_subplot(1, 2, 2)]
	axs[0].imshow(img)
	axs[1].imshow(cv2.merge((result_rsf,result_rsf,result_rsf)))
	plt.show()
	#
	#img_rsf = cv2.merge((result_rcf, result_rcf, result_rcf))
	#cv2.imwrite(Path_img_edges, img_rsf)
	exit()
