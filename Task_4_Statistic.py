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

Root_dir = "/media/kolad/HardDisk/Zirkon"

Path_dir = "/home/kolad/PycharmProjects/ZirkonProcessing/temp"
FileNames = os.listdir(Path_dir + "/Data/")


for FileName in FileNames:
       print(FileName)
       mat = loadmat(Path_dir + "/Data/" + FileName, squeeze_me=True)
       Weibull_mat = loadmat(Path_dir + "/Data_Weibull/" + "Weibull_ks_" + FileName[0:4] + ".mat",
                             squeeze_me=True)
       LogNorm_mat = loadmat(Path_dir + "/Data_Log-Norm/" + "Log-Norm_ks_" + FileName[0:4] + ".mat",
                             squeeze_me=True)
       Power_mat = loadmat(Path_dir + "/Data_Power/" + "Power_ks_" + FileName[0:4] + ".mat",
                             squeeze_me=True)

       ImgName = FileName[0:FileName.find('_')]
       RootImgDir = Root_dir + "/ZirkonChoosen"
       EdgeImgDir =  Root_dir + "/ZirkonUpscaleBINEdgesPrep"
       SegImgDir = Root_dir + "/ZirkonUpscaleSegmentation"
       RootImgFile = None
       EdgeImgFile = None
       def GetFileName(Name, Dir):
              for file in os.listdir(Dir):
                     if file.find(Name) != -1:
                            return file



       img_root = cv2.cvtColor(cv2.imread(RootImgDir + "/" + GetFileName(ImgName, RootImgDir)), cv2.COLOR_BGR2RGB)
       img_edge = cv2.cvtColor(cv2.imread(EdgeImgDir + "/" + GetFileName(ImgName, EdgeImgDir)), cv2.COLOR_BGR2RGB)
       img_seg = cv2.cvtColor(cv2.imread(SegImgDir + "/" + GetFileName(ImgName, SegImgDir)), cv2.COLOR_BGR2RGB)

       #
       S = mat['S']
       del mat
       xmin = np.min(S)
       xmax = np.max(S)
       n = np.ceil(np.log2(xmax/xmin))
       f_bins = xmin*np.logspace(0,n,15,base=2)
       S = S[(S >= xmin) & (S <= xmax)]

       dist = st.lognorm(LogNorm_mat['sigma'], 0, LogNorm_mat['lambda'])
       LogNorm_mat['F'] = (dist.cdf(f_bins) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))

       F2 = lambda x: SE.Fweibull(x, Weibull_mat['alpha'], Weibull_mat['lambda'])
       Weibull_mat['F'] = (F2(f_bins) - F2(xmin)) / (F2(xmax) - F2(xmin))

       F3 = lambda x: SE.Fparetomodif(x, Power_mat['alpha'], Power_mat['lambda'])
       Power_mat['F'] = (F3(f_bins) - F3(xmin)) / (F3(xmax) - F3(xmin))

       Sx, Fx = get_ecdf(S, xmin=xmin)

       alpha = 0.25
       Power_mat['q'] = np.quantile(Power_mat['ks'], 1 - alpha)
       Weibull_mat['q'] = np.quantile(Weibull_mat['ks'], 1 - alpha)
       Power_mat['q'] = np.quantile(Power_mat['ks'], 1 - alpha)

       fig = plt.figure(figsize=(14, 9))
       axs = [fig.add_subplot(2, 3, 1),
              fig.add_subplot(2, 3, 2),
              fig.add_subplot(2, 3, 3),
              fig.add_subplot(2, 3, 4),
              fig.add_subplot(2, 3, 5),
              fig.add_subplot(2, 3, 6)]
       #
       axs[0].imshow(img_root)
       axs[0].set_title('a)')
       #
       axs[1].imshow(img_edge)
       axs[1].set_title('b)')
       #
       axs[2].imshow(img_seg)
       axs[2].set_title('c)')
       #
       axs[3].fill_between(f_bins, LogNorm_mat['F']-Power_mat['q'], LogNorm_mat['F']+Power_mat['q'],
                           alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
       axs[3].plot(f_bins, LogNorm_mat['F'], color='black', label="Log-normal law")
       axs[3].plot(Sx, Fx, color='red', label="Empirical cdf")
       axs[3].set_xscale('log')
       axs[3].set_xlim([xmin, xmax])
       axs[3].set_ylim([0, 1])
       axs[3].set_title('d)')
       axs[3].legend(loc='lower right')
       axs[3].set_xlabel('-, p.u.')
       #
       axs[4].fill_between(f_bins, Weibull_mat['F'] - Weibull_mat['q'], Weibull_mat['F'] + Weibull_mat['q'],
                           alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
       axs[4].plot(f_bins, Weibull_mat['F'], color='black', label="Weibull law")
       axs[4].plot(Sx,Fx, color='red', label="Empirical cdf")
       axs[4].set_xscale('log')
       axs[4].set_xlim([xmin, xmax])
       axs[4].set_ylim([0, 1])
       axs[4].set_title('e)')
       axs[4].legend(loc='lower right')
       axs[4].set_xlabel('-, p.u.')
       #
       axs[5].fill_between(f_bins, Power_mat['F'] - Power_mat['q'], Power_mat['F'] + Power_mat['q'],
                           alpha=0.6, linewidth=0, color='grey', label="Сonfidence interval")
       axs[5].plot(f_bins, Power_mat['F'], color='black', label="Power law")
       axs[5].plot(Sx, Fx, color='red', label="Empirical cdf")
       axs[5].set_xscale('log')
       axs[5].set_xlim([xmin, xmax])
       axs[5].set_ylim([0, 1])
       axs[5].set_title('f)')
       axs[5].legend(loc='lower right')
       axs[5].set_xlabel('-, p.u.')
       #
       fig.suptitle(ImgName, fontsize=16)
       fig.savefig("temp/Pictures/" + ImgName + ".png")
       #plt.show()
       #exit()
       plt.close('all')


exit()