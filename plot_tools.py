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


def plot_data(data):
    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)
    _ecdf = data["ecdf"]
    x = data["x"]
    xmin = data["xmin"]
    xmax = data["xmax"]

    fig = plt.figure(figsize=(12, 4))
    axs = [fig.add_subplot(1, 3, 1),
           fig.add_subplot(1, 3, 2),
           fig.add_subplot(1, 3, 3)]
    plot_distribution(axs[0], data, "lognorm")
    plot_distribution(axs[1], data, "weibull")
    plot_distribution(axs[2], data, "paretoexp")
    plt.subplots_adjust(bottom=0.2)
    plt.show()


def plot_distribution(ax, data, name):
    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)

    ecdf = data["ecdf"]
    x = data["x"]
    xmin = data["xmin"]
    xmax = data["xmax"]

    test_data = data["test_data"][name]
    cdf_min = test_data["cdf_min"]
    cdf_max = test_data["cdf_max"]
    cdf = test_data["cdf"]

    ax.fill_between(x, cdf_min, cdf_max, color="gray", label="1")
    ax.plot(x, cdf, color="black", linestyle="--", label="2")
    ax.plot(ecdf["values"], ecdf["freqs"], color="black", label="3")
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