import math
import os
import numpy as np
import cv2
from scipy.io import loadmat

from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt
import StatisticEstimation as SE
from StatisticEstimation import GetThetaLognorm
from scipy import stats as st
from StatisticEstimation import get_ecdf, lcdfgen, get_pv
from scipy.io import savemat

Path_dir = "/home/kolad/PycharmProjects/ZirkonProcessing/temp/Data"
FileNames = os.listdir(Path_dir)


for FileName in FileNames:
       print(FileName)
       mat = loadmat(Path_dir + "/" + FileName, squeeze_me=True)
       S = mat['S']
       del mat
       xmin = np.min(S)
       xmin = max(xmin, 20)
       xmax = np.max(S)
       S = S[(S >= xmin) & (S <= xmax)]
       xmin = np.min(S)

       n = np.ceil(np.log2(xmax/xmin))
       f_bins = xmin*np.logspace(0,n,15,base=2)
       f, _ = np.histogram(S, bins=f_bins, density=True)


       theta = SE.GetThetaWeibull(S, xmin, xmax)
       N = len(S)
       Sx0, Fx0 = get_ecdf(S)
       k_alpha = theta[0]
       lam = theta[2]
       F = lambda x: SE.Fweibull(x, theta[0], lam)
       F_log = (F(f_bins) - F(xmin))/(F(xmax) - F(xmin))
       lF0 = (F(Sx0) - F(xmin)) / (F(xmax) - F(xmin))
       ks0 = np.max(np.abs(lF0 - Fx0))

       ac = 1000
       ks1 = np.empty(ac)
       ks2 = np.empty(ac)
       for i in range(ac):
              if i % 100 == 0:
                     print(i, '/', ac)
              # Виртуальный элемент ансамбля
              s = lcdfgen(Sx0, Fx0, N)
              Sx, Fxp = get_ecdf(s, xmin=xmin)
              theta = SE.GetThetaWeibull(s, xmin, xmax)
              F = lambda x: SE.Fweibull(x, theta[0], theta[2])
              # Аппроксимация виртуального элемента
              lF = (F(Sx) - F(xmin)) / (F(xmax) - F(xmin))
              ks1[i] = np.max(np.abs(Fxp - lF))
              # элемент ансамбля оценки виртуального элмента
              s = lcdfgen(Sx, lF/np.max(lF), N)
              Sx, Fxp = get_ecdf(s)
              lF = (F(Sx) - F(xmin)) / (F(xmax) - F(xmin))
              ks2[i] = np.max(np.abs(Fxp - lF))

       pv = get_pv(ks2, ks0)

       save_dict = {'ks':ks2,'alpha':k_alpha,'lambda':lam, 'pv': pv, 'N': N}
       savemat("temp/Data_Weibull/Weibull_ks_" + FileName[0:4] + ".mat", save_dict)

       # ===============
       alpha = 0.25
       q1 = np.quantile(ks2, alpha)
       q2 = np.quantile(ks2, 1 - alpha)
       F_min = F_log - q2
       F_max = F_log + q2

       fig = plt.figure(figsize=(14, 9))
       axs = [fig.add_subplot(1, 1, 1)]
       axs[0].fill_between(f_bins, F_min, F_max,
                           alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
       axs[0].plot(f_bins,F_log, color='black')
       axs[0].plot(Sx0, Fx0, color='red')
       axs[0].set_xscale('log')
       axs[0].set_xlim([xmin, xmax])
       axs[0].set_ylim([0, 1])
       fig.suptitle(FileName + ' Weilbur', fontsize=16)
       fig.savefig("temp/Pictures_Weibull/" + FileName + "_S.png")
       plt.close('all')

