import math
import os
import numpy as np
import cv2
from scipy.io import loadmat

from rsf_edges import modelini, get_model_edges, modelgpu
import matplotlib.pyplot as plt
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


for FileName in FileNames:
       print(FileName)
       mat = loadmat(Path_dir + "/" + FileName, squeeze_me=True)
       S = mat['S']
       del mat

       xmin = np.min(S)
       xmax = np.max(S)
       n = np.ceil(np.log2(xmax/xmin))
       f_bins = xmin*np.logspace(0,n,15,base=2)
       S = S[(S >= xmin) & (S <= xmax)]
       f, _ = np.histogram(S, bins=f_bins, density=True)


       theta = GetThetaLognorm(S, xmin, xmax)
       dist = st.lognorm(theta[0], 0, theta[2])
       fx = (f_bins[0:-1] + f_bins[1:])/2
       f_log = dist.pdf(fx)/(dist.cdf(xmax) - dist.cdf(xmin))

       N = len(S)
       Sx, Fx = get_ecdf(S)
       S2 = lcdfgen(Sx,Fx, N)
       Sx2, Fx2 = get_ecdf(S2)



       F_log = (dist.cdf(f_bins)-dist.cdf(xmin))/(dist.cdf(xmax) - dist.cdf(xmin))


       ac = 1000
       ks1 = np.empty(ac)
       ks2 = np.empty(ac)
       for i in range(ac):
              if i % 100 == 0:
                     print(i, '/', ac)
              s = lcdfgen(Sx,Fx, N)
              Sx, Fxp = get_ecdf(s)
              theta = GetThetaLognorm(s, xmin, xmax)
              dist = st.lognorm(theta[0], 0, theta[2])
              lF = (dist.cdf(Sx) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))
              ks1[i] = np.max(np.abs(Fxp - lF))
              s = lcdfgen(Sx, lF, N)
              Sx, Fxp = get_ecdf(s)
              lF = (dist.cdf(Sx) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))
              ks2[i] = np.max(np.abs(Fxp - lF))



       bins = np.linspace(np.min(ks), np.max(ks), 15)
       fig = plt.figure(figsize=(14, 9))
       axs = [fig.add_subplot(1, 1, 1)]
       axs[0].hist(ks1, bins=bins)
       axs[0].hist(ks2, bins=bins)
       #fig.suptitle(FileName, fontsize=16)
       plt.show()

       exit()

#
#
#
#        q = 0.5
#
#        # =======
#        f_max = np.quantile(af, 1-q, axis=0)
#        f_min = np.quantile(af, q, axis=0)
#        f_max = np.insert(f_max, 0, f_max[0])
#        f_min = np.append(f_min, f_min[-1])
#        # ==
#        F_max = np.quantile(aF, 1-q, axis=0)
#        F_min = np.quantile(aF, q, axis=0)
#        # =======
#
#        fig = plt.figure(figsize=(14, 9))
#        axs = [fig.add_subplot(1, 2, 1),
#               fig.add_subplot(1, 2, 2)]
#        axs[0].fill_between(f_bins, F_min, F_max,
#                            alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
#        axs[0].plot(f_bins,F_log, color='black')
#        axs[0].plot(Sx,Fx, color='red')
#        axs[0].set_xscale('log')
#        axs[0].set_xlim([xmin, xmax])
#
#        axs[1].fill_between(f_bins, f_min, f_max,
#                            alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
#        axs[1].plot(fx,f_log,color='black')
#        axs[1].set_xscale('log')
#        axs[1].set_yscale('log')
#        axs[1].set_xlim([f_bins[1], f_bins[-2]])
#        fig.suptitle(FileName, fontsize=16)
#        fig.savefig("temp/" + FileName + "_S.png")
#        plt.close('all')
#        #plt.show()
#        #exit()
#
# exit()