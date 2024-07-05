import math
import os
import numpy as np
import cv2
from scipy.io import loadmat

from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt
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


       theta = GetThetaLognorm(S, xmin, xmax)
       dist = st.lognorm(theta[0], 0, theta[2])

       sigma = theta[0]
       lam = theta[2]

       fx = (f_bins[0:-1] + f_bins[1:])/2
       f_log = dist.pdf(fx)/(dist.cdf(xmax) - dist.cdf(xmin))

       N = len(S)
       Sx0, Fx0 = get_ecdf(S)
       F_log = (dist.cdf(f_bins)-dist.cdf(xmin))/(dist.cdf(xmax) - dist.cdf(xmin))
       lF0 = (dist.cdf(Sx0) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))
       ks0 = np.max(np.abs(lF0 - Fx0))


       ac = 1000
       ks1 = np.empty(ac)
       ks2 = np.empty(ac)
       for i in range(ac):
              if i % 100 == 0:
                     print(i, '/', ac)
              # Виртуальный элемент ансамбля
              s = lcdfgen(Sx0, Fx0, N)
              Sx, Fxp = get_ecdf(s, xmin)
              theta = GetThetaLognorm(s, xmin, xmax)
              dist = st.lognorm(theta[0], 0, theta[2])
              # Аппроксимация виртуального элемента
              lF = (dist.cdf(Sx) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))
              ks1[i] = np.max(np.abs(Fxp - lF))
              # элемент ансамбля оценки виртуального элмента
              s = lcdfgen(Sx, lF/np.max(lF), N)
              Sx, Fxp = get_ecdf(s)
              lF = (dist.cdf(Sx) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))
              ks2[i] = np.max(np.abs(Fxp - lF))


       pv = get_pv(ks2, ks0)

       print(pv)
       save_dict = {'ks': ks2, 'sigma': sigma, 'lambda': lam, 'pv': pv, 'N': N}
       savemat("temp/Data_Log-Norm/Log-Norm_ks_" + FileName[0:4] + ".mat", save_dict)

       # ============
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
       fig.suptitle(FileName + " lognorm", fontsize=16)
       fig.savefig("temp/Pictures_Log-Norm/" + FileName + "_S.png")
       #plt.show()
       plt.close('all')
       #exit()
