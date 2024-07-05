import math
import os
import numpy as np
import cv2
from scipy.io import loadmat

from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt
from StatisticEstimation import GetThetaLognorm
from scipy import stats as st
from StatisticEstimation import get_ecdf, lcdfgen
import StatisticEstimation as SE
import matplotlib as mpl
import pandas as pd

import matplotlib.font_manager

font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
custom_font = mpl.font_manager.FontProperties(fname=font_path, size = 16)


Root_dir = "/media/kolad/HardDisk/Zirkon"

Path_dir = "/home/kolad/PycharmProjects/ZirkonProcessing/temp"
FileNames = os.listdir(Path_dir + "/Data/")


pv = {"LogNorm": np.empty(len(FileNames)),
      "Weibull": np.empty(len(FileNames)),
      "Power": np.empty(len(FileNames))}

N = np.empty(len(FileNames))
i = 0
for FileName in FileNames:
       print(FileName)
       mat = loadmat(Path_dir + "/Data/" + FileName, squeeze_me=True)
       Weibull_mat = loadmat(Path_dir + "/Data_Weibull/" + "Weibull_ks_" + FileName[0:4] + ".mat",
                             squeeze_me=True)
       LogNorm_mat = loadmat(Path_dir + "/Data_Log-Norm/" + "Log-Norm_ks_" + FileName[0:4] + ".mat",
                             squeeze_me=True)
       Power_mat = loadmat(Path_dir + "/Data_Power/" + "Power_ks_" + FileName[0:4] + ".mat",
                             squeeze_me=True)


       pv["LogNorm"][i] = LogNorm_mat["pv"]
       pv["Power"][i] = Power_mat["pv"]
       pv["Weibull"][i] = Weibull_mat["pv"]
       N[i] = LogNorm_mat["N"]
       i = i + 1
print(pv["LogNorm"])
print(pv["Power"])
print(pv["Weibull"])
print(FileNames)

df = pd.DataFrame({
    "Names": FileNames,
    "LogNorm": 1-pv["LogNorm"],
    "Power": 1-pv["Power"],
    "Weibull": 1-pv["Weibull"]})
print(np.median(1-pv["LogNorm"]))
print(np.median(1-pv["Power"]))
print(np.median(1-pv["Weibull"]))
exit()
fig = plt.figure(figsize=(14, 9))
axs = [fig.add_subplot(1, 1, 1)]
axs[0].plot(N, 1-pv["LogNorm"], '.','--',color='black')
axs[0].plot(N, 1-pv["Power"], 'o','--',color='black')
axs[0].plot(N, 1-pv["Weibull"], '*','--',color='black')
axs[0].plot([np.min(N)-5, np.max(N)+5], [0.95 ,0.95],'--',color='black')
axs[0].plot([np.min(N)-5, np.max(N)+5], [0.05 ,0.05],'--',color='black')

axs[0].plot([np.min(N)-5, np.max(N)+5], [0.75 ,0.75],'--',color='black')
axs[0].plot([np.min(N)-5, np.max(N)+5], [0.25 ,0.25],'--',color='black')
axs[0].set_xlim([np.min(N)-5, np.max(N)+5])
plt.show()
exit()