import math
import os
import numpy as np
import cv2
from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt


class RCF_overcrop:
	def get_crop_edge(self, x, y, dx, dy, ddx, ddy):
		img_small = self.img[y:y + dy, x:x + dx]
		self.result_rsf[y+ddy:y+dy-ddy, x+ddx:x+dx-ddx] = self.model.get_model_edges(img_small)[ddy:dy-ddy, ddx:dx-ddx]

	def get_full_edge(self, dx, dy, ddx, ddy):
		jmax = math.floor((self.sh[0] - 2 * ddy) / (dy - 2 * ddy))
		imax = math.floor((self.sh[1] - 2 * ddx) / (dx - 2 * ddx))
		for i in range(0, imax):
			for j in range(0, jmax):
				if (i*jmax + j) % 5 == 0:
					print(i*jmax + j,"/",imax*jmax)
				self.get_crop_edge((dx - 2 * ddx) * i, (dy - 2 * ddy) * j, dx, dy, ddx, ddy)
			y = (dy - 2 * ddy) * jmax
			self.get_crop_edge((dx - 2 * ddx) * i, y, dx, self.sh[0] - y, ddx, ddy)
		for j in range(0, jmax):
			x = (dx - 2 * ddx) * imax
			self.get_crop_edge(x, (dy - 2 * ddy) * j, self.sh[1] - x, dy, ddx, ddy)
		return self.result_rsf

	def __init__(self,img):
		self.img = img
		self.sh = img.shape
		self.model = modelgpu()
		self.result_rsf = np.zeros(img.shape[0:2], np.float32)


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
ddx = 64
ddy = 64
Path_dir = "/media/kolad/HardDisk/Zirkon"

FileNames = os.listdir(Path_dir)
print(FileNames)


FileName1 = "Z-20e.png"
FileName2 = "Z-20eup8.png"



Path_img1 = Path_dir + "/" + FileName1
Path_img2 = Path_dir + "/" + FileName2
img1 = cv2.imread(Path_img1)
img2 = cv2.imread(Path_img2)
model = modelgpu()
result_rsf1 = model.get_model_edges(img1)
result_rsf2 = RCF_overcrop(img2).get_full_edge(dx, dy, ddx, ddy)

result_rsf1 = np.uint8((result_rsf1 / result_rsf1.max()) * 255)
result_rsf2 = np.uint8((result_rsf2 / result_rsf2.max()) * 255)


result_rsf1 = step_skeleton(result_rsf1)
#result_rsf2 = step_skeleton(result_rsf2)
ret, result_rsf2 = cv2.threshold(result_rsf2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


cv2.imwrite(Path_dir + "/" + FileName1[:-3] + "_edge_0.png", result_rsf1)
cv2.imwrite(Path_dir + "/" + FileName2[:-3] + "_edge_0.png", result_rsf2)

fig = plt.figure(figsize=(14, 9))
axs = [fig.add_subplot(2, 2, 1),
       fig.add_subplot(2, 2, 2),
       fig.add_subplot(2, 2, 3),
       fig.add_subplot(2, 2, 4)]
axs[0].imshow(img1)
axs[1].imshow(result_rsf1)
axs[2].imshow(img2)
axs[3].imshow(result_rsf2)

plt.show()

exit()
