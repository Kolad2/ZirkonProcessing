import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
import json
from pathlib import Path
from tqdm import tqdm
from pyrockstats import ecdf
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from pyrockstats.bootstrap.ks_statistics import get_ks_distribution
from pyrockstats.bootstrap.ks_statistics import get_confidence_value
from plot_tools import plot_data
from distrebution_test import DistributionTest

def main():
    with open("./data/zirkons_areas.json") as file:
        data = json.load(file)
    out_data = {}
    for name in data:
        print(name + " " + "start")
        _out_data = get_test_data(data[name]["areas"])
        plot_data(_out_data)
        out_data[name] = _out_data
        print(name + " " + "ready")

    with open("./data/zirkons_tests.json", 'w+') as json_file:
        json.dump(out_data, json_file, indent=4)


def get_test_data(areas):
    areas = np.array(areas)
    models = {
        "lognorm": lognorm,
        "weibull": weibull,
        "paretoexp": paretoexp
    }
    xmin = np.min(areas)
    xmax = np.max(areas)

    values, e_freq = ecdf(areas)
    tests = {
        name: DistributionTest(areas, model) for name, model in models.items()
    }

    x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    alpha = 0.05
    data = {
        "x": x.tolist(),
        "xmin": xmin,
        "xmax": xmax,
        "alpha": alpha,
        "test_data": {name: test.get_data(x, alpha) for name, test in tests.items()},
        "ecdf": {"values": values.tolist(), "freqs": e_freq.tolist()}
    }
    return data



if __name__ == '__main__':
    main()



# img_root = cv2.cvtColor(cv2.imread(RootImgDir + "/" + GetFileName(ImgName, RootImgDir)), cv2.COLOR_BGR2RGB)
        # img_edge = cv2.cvtColor(cv2.imread(EdgeImgDir + "/" + GetFileName(ImgName, EdgeImgDir)), cv2.COLOR_BGR2RGB)
        # img_seg = cv2.cvtColor(cv2.imread(SegImgDir + "/" + GetFileName(ImgName, SegImgDir)), cv2.COLOR_BGR2RGB)
        #
        # #
        # S = mat['S']
        # del mat
        # xmin = np.min(S)
        # xmax = np.max(S)
        # n = np.ceil(np.log2(xmax / xmin))
        # f_bins = xmin * np.logspace(0, n, 15, base=2)
        # S = S[(S >= xmin) & (S <= xmax)]
        #
        # dist = st.lognorm(LogNorm_mat['sigma'], 0, LogNorm_mat['lambda'])
        # LogNorm_mat['F'] = (dist.cdf(f_bins) - dist.cdf(xmin)) / (dist.cdf(xmax) - dist.cdf(xmin))
        #
        # F2 = lambda x: SE.Fweibull(x, Weibull_mat['alpha'], Weibull_mat['lambda'])
        # Weibull_mat['F'] = (F2(f_bins) - F2(xmin)) / (F2(xmax) - F2(xmin))
        #
        # F3 = lambda x: SE.Fparetomodif(x, Power_mat['alpha'], Power_mat['lambda'])
        # Power_mat['F'] = (F3(f_bins) - F3(xmin)) / (F3(xmax) - F3(xmin))
        #
        # Sx, Fx = get_ecdf(S, xmin=xmin)
        #
        # alpha = 0.25
        # Power_mat['q'] = np.quantile(Power_mat['ks'], 1 - alpha)
        # Weibull_mat['q'] = np.quantile(Weibull_mat['ks'], 1 - alpha)
        # Power_mat['q'] = np.quantile(Power_mat['ks'], 1 - alpha)
        #
        # fig = plt.figure(figsize=(14, 9))
        # axs = [fig.add_subplot(2, 3, 1),
        #        fig.add_subplot(2, 3, 2),
        #        fig.add_subplot(2, 3, 3),
        #        fig.add_subplot(2, 3, 4),
        #        fig.add_subplot(2, 3, 5),
        #        fig.add_subplot(2, 3, 6)]
        # #
        # axs[0].imshow(img_root)
        # axs[0].set_title('а)', fontsize=16, fontproperties=custom_font)
        # axs[0].get_xaxis().set_visible(False)
        # axs[0].get_yaxis().set_visible(False)
        # #
        # axs[1].imshow(img_edge)
        # axs[1].set_title('б)', fontsize=16, fontproperties=custom_font)
        # axs[1].get_xaxis().set_visible(False)
        # axs[1].get_yaxis().set_visible(False)
        # #
        # axs[2].imshow(img_seg)
        # axs[2].set_title('в)', fontsize=16, fontproperties=custom_font)
        # axs[2].get_xaxis().set_visible(False)
        # axs[2].get_yaxis().set_visible(False)
        # #
        # axs[3].fill_between(f_bins * (0.043 ** 2), LogNorm_mat['F'] - Power_mat['q'], LogNorm_mat['F'] + Power_mat['q'],
        #                     alpha=0.6, linewidth=0, color='grey', label="1")
        # axs[3].plot(f_bins * (0.043 ** 2), LogNorm_mat['F'], color='black', label="2", linestyle="--")
        # axs[3].plot(Sx * (0.043 ** 2), Fx, color='black', label="3")
        # axs[3].set_xscale('log')
        # axs[3].set_xlim([xmin * (0.043 ** 2), xmax * (0.043 ** 2)])
        # axs[3].set_ylim([0, 1])
        # axs[3].set_title('а)', fontsize=16, fontproperties=custom_font)
        # axs[3].legend(loc='lower right', fontsize=16, prop=custom_font)
        # axs[3].set_xlabel(r'S, мкм$^\mathregular{2}$', fontproperties=custom_font)
        # axs[3].xaxis.set_tick_params(labelsize=16)
        # axs[3].yaxis.set_tick_params(labelsize=16, labelfontfamily='sans-serif')
        # #
        # axs[4].fill_between(f_bins * (0.043 ** 2), Weibull_mat['F'] - Weibull_mat['q'],
        #                     Weibull_mat['F'] + Weibull_mat['q'],
        #                     alpha=0.6, linewidth=0, color='grey', label="1")
        # axs[4].plot(f_bins * (0.043 ** 2), Weibull_mat['F'], color='black', label="2", linestyle="--")
        # axs[4].plot(Sx * (0.043 ** 2), Fx, color='black', label="3")
        # axs[4].set_xscale('log')
        # axs[4].set_xlim([xmin * (0.043 ** 2), xmax * (0.043 ** 2)])
        # axs[4].set_ylim([0, 1])
        # axs[4].set_title('б)', fontsize=16, fontproperties=custom_font)
        # axs[4].legend(loc='lower right', fontsize=16, prop=custom_font)
        # axs[4].set_xlabel(r'S, мкм$^\mathregular{2}$', fontproperties=custom_font)
        # axs[4].xaxis.set_tick_params(labelsize=16)
        # axs[4].yaxis.set_tick_params(labelsize=16)
        # #
        # axs[5].fill_between(f_bins * (0.043 ** 2), Power_mat['F'] - Power_mat['q'], Power_mat['F'] + Power_mat['q'],
        #                     alpha=0.6, linewidth=0, color='grey', label="1")
        # axs[5].plot(f_bins * (0.043 ** 2), Power_mat['F'], color='black', label="2", linestyle="--")
        # axs[5].plot(Sx * (0.043 ** 2), Fx, color='black', label="3")
        # axs[5].set_xscale('log')
        # axs[5].set_xlim([xmin * (0.043 ** 2), xmax * (0.043 ** 2)])
        # axs[5].set_ylim([0, 1])
        # axs[5].set_title('в)', fontproperties=custom_font)
        # axs[5].legend(loc='lower right', prop=custom_font)
        # axs[5].set_xlabel(r'S, мкм$^\mathregular{2}$', fontproperties=custom_font)
        # # Установка шрифта для чисел делений шкал
        # for ax in axs:
        #     for label in ax.get_xticklabels() + ax.get_yticklabels():
        #         label.set_fontproperties(custom_font)
        # #
        # fig.suptitle(ImgName, fontsize=16)
        # fig.savefig("temp/Pictures/" + ImgName + ".png")
        # plt.show()
        # plt.close('all')