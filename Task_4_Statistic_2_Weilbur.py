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

def get_ecdf(X):
       X, C = np.unique(X, return_counts=True)
       C[0] = 0
       F = np.cumsum(C)
       F = F/F[-1]
       return X, F

def lcdfgen(X, F, N):
       Fx = np.sort(np.random.rand(N))
       x = np.zeros(np.shape(Fx))
       k = 1
       for i in range(N):
              while Fx[i] > F[k]:
                     k = k + 1
              x[i] = (X[k] - X[k-1])/(F[k] - F[k-1])*(Fx[i] - F[k-1]) + X[k-1]
       return x

Path_dir = "/home/kolad/PycharmProjects/ZirkonProcessing/temp/Data"
FileNames = os.listdir(Path_dir)


for FileName in FileNames[1:]:
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
       F = lambda x: SE.Fweibull(x, theta[0], theta[2])
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
              Sx, Fxp = get_ecdf(s)
              theta = SE.GetThetaWeibull(s, xmin, xmax)
              F = lambda x: SE.Fweibull(x, theta[0], theta[2])
              # Аппроксимация виртуального элемента

              lF = (F(Sx) - F(xmin)) / (F(xmax) - F(xmin))
              ks1[i] = np.max(np.abs(Fxp - lF))
              # элемент ансамбля оценки виртуального элмента
              s = lcdfgen(Sx, lF/np.max(lF), N)
              Sx, Fxp = get_ecdf(s)
              print(Sx)
              lF = (F(Sx) - F(xmin)) / (F(xmax) - F(xmin))
              ks2[i] = np.max(np.abs(Fxp - lF))

       alpha = 0.1
       print(ks2)
       q1 = np.quantile(ks2, alpha)
       q2 = np.quantile(ks2, 1 - alpha)
       print(q1,q2,ks0)



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
       fig.suptitle(FileName, fontsize=16)
       fig.savefig("temp/" + FileName + "_S.png")
       #plt.show()
       plt.close('all')
       #exit()
