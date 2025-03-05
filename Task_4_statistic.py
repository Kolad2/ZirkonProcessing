import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager
import json
from pathlib import Path

from pyrockstats import ecdf
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from pyrockstats.bootstrap.ks_statistics import get_ks_distribution
from pyrockstats.bootstrap.ks_statistics import get_confidence_value


def main():
    with open("./data/zirkons_areas.json") as file:
        data = json.load(file)
    
    for name in data:
        print(name)
        fun1(data[name]["areas"])
        exit()


def fun1(areas):
    models = {
        "lognorm": lognorm,
        "weibull": weibull,
        "paretoexp": paretoexp
    }

    areas = np.array(areas)
    xmin = np.min(areas)
    xmax = np.max(areas)

    values, e_freq = ecdf(areas)
    tests = {
        name: DistributionTest(areas, model) for name, model in models.items()
    }

    x = np.logspace(np.log10(xmin), np.log10(xmax), 100)
    alpha = 0.05
    plot_datas = {name: test.get_plot_data(x, alpha) for name, test in tests.items()}


    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)

    def plot_distribution(ax, plot_data):
        cdf_min = plot_data["cdf_min"]
        cdf_max = plot_data["cdf_max"]
        cdf = plot_data["cdf"]

        ax.fill_between(x, cdf_min, cdf_max, color="gray", label="1")
        ax.plot(x, cdf, color="black", linestyle="--", label="2")
        ax.plot(values, e_freq, color="black", label="3")
        ax.set_xscale('log')
        ax.set_xlabel(r's, мкм$^\mathregular{2}$', fontproperties=custom_font, size=16)
        for label in ax.get_xticklabels():
            label.set_fontproperties(custom_font)
            label.set_size(16)
        for label in ax.get_yticklabels():
            label.set_fontproperties(custom_font)
            label.set_size(16)
        ax.legend(loc='lower right', fontsize=16, prop=custom_font)
        ax.set_ylim([0, 1])
        ax.set_xlim([xmin, xmax])

    fig = plt.figure(figsize=(12, 4))
    axs = [fig.add_subplot(1, 3, 1),
           fig.add_subplot(1, 3, 2),
           fig.add_subplot(1, 3, 3)]
    plot_distribution(axs[0], plot_datas["lognorm"])
    plot_distribution(axs[1], plot_datas["weibull"])
    plot_distribution(axs[2], plot_datas["paretoexp"])
    plt.subplots_adjust(bottom=0.2)
    plt.show()

class DistributionTest:
    def __init__(self, areas, model):
        self.xmin = np.min(areas)
        self.xmax = np.max(areas)
        self.model = model
        self.ks = get_ks_distribution(areas, model, n_ks=500)
        self.theta = self.model.fit(areas, xmin=self.xmin, xmax=self.xmax)
        self.dist = self.model(*self.theta, xmin=self.xmin, xmax=self.xmax)
        self.confidence_value = None
        self.alpha = None

    def get_confidence_value(self, alpha):
        if self.alpha is not None and alpha == self.alpha:
            return self.confidence_value
        self.alpha = alpha
        self.confidence_value = get_confidence_value(self.ks, significance=alpha)
        return self.confidence_value

    def model_cdf(self, x):
        print(x)
        return self.dist.cdf(x, xmin=self.xmin, xmax=self.xmax)

    def get_plot_data(self, x, alpha):
        confidence_value = self.get_confidence_value(alpha)
        cdf = self.model_cdf(x)
        cdf_min = cdf - confidence_value
        cdf_max = cdf + confidence_value
        plot_data = {
            "cdf": cdf,
            "cdf_min": cdf_min,
            "cdf_max": cdf_max
        }
        return plot_data


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