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
       F = np.cumsum(C)
       F = F/F[-1]
       return X, F

def lcdfgen(X, F, N):
       Fx = np.sort(np.random.rand(N))
       x = np.zeros(np.shape(Fx))
       k = 0
       for i in range(N):
              while Fx[i] > F[k]:
                     k = k + 1
              x[i] = X[k]
       return x



mat = loadmat("temp/S.mat", squeeze_me=True)
S = mat['S']
del mat

xmin = 8*5
f_bins = xmin*np.logspace(0,12,15,base=2)
xmax = f_bins[-1]
S = S[(S > xmin) & (S < xmax)]
f, _ = np.histogram(S, bins=f_bins, density=True)


theta = GetThetaLognorm(S, xmin, xmax)
dist = st.lognorm(theta[0], 0, theta[2])
fx = (f_bins[0:-1] + f_bins[1:])/2
f_log = dist.pdf(fx)/(1 - dist.cdf(xmin))



N = len(S)
Sx, Fx = get_ecdf(S)
S2 = lcdfgen(Sx,Fx, N)
Sx2, Fx2 = get_ecdf(S2)


F_log = (dist.cdf(f_bins)-dist.cdf(xmin))/(1 - dist.cdf(xmin))


ac = 10000
af = np.empty((ac,len(f_bins)-1))
aF = np.empty((ac,len(f_bins)))
for i in range(ac):
       s = lcdfgen(Sx,Fx, N)
       af[i], _ = np.histogram(s, bins=f_bins, density=True)
       sxp, Fxp = get_ecdf(s)
       aF[i] = np.interp(f_bins, sxp, Fxp)

q = 0.05

# =======
f_max = np.quantile(af, 1-q, axis=0)
f_min = np.quantile(af, q, axis=0)
f_max = np.insert(f_max, 0, f_max[0])
f_min = np.append(f_min,f_min[-1])
# ==
F_max = np.quantile(aF, 1-q, axis=0)
F_min = np.quantile(aF, q, axis=0)
# =======

fig = plt.figure(figsize=(14, 9))
axs = [fig.add_subplot(1, 2, 1),
       fig.add_subplot(1, 2, 2)]
axs[0].fill_between(f_bins, F_min, F_max,
                    alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
axs[0].plot(f_bins,F_log, color='black')
axs[0].plot(Sx,Fx, color='red')
axs[0].set_xscale('log')
axs[0].set_xlim([Sx[0], Sx[-1]])

axs[1].fill_between(f_bins, f_min, f_max,
                    alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
axs[1].plot(fx,f_log,color='black')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlim([f_bins[1], f_bins[-2]])
plt.show()

exit()