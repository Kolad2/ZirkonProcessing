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
from pyrockstats.distrebutions import lognorm, weibull, paretoexp
from pyrockstats.bootstrap.ks_statistics import get_ks_distribution
from pyrockstats.bootstrap.ks_statistics import get_confidence_value


def main():
    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)

    with open("./data/zirkons_tests.json") as file:
        full_data = json.load(file)

    for name, data in full_data.items():
        print(name)
        plot_data(data)
        exit()

def plot_data(data):
    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)
    _ecdf = data["ecdf"]
    x = data["x"]
    xmin = data["xmin"]
    xmax = data["xmax"]

    from plot_tools import plot_distribution

    fig = plt.figure(figsize=(12, 4))
    axs = [fig.add_subplot(1, 3, 1),
           fig.add_subplot(1, 3, 2),
           fig.add_subplot(1, 3, 3)]
    plot_distribution(axs[0], data, "lognorm")
    plot_distribution(axs[1], data, "weibull")
    plot_distribution(axs[2], data, "paretoexp")
    plt.subplots_adjust(bottom=0.2)
    plt.show()

if __name__ == '__main__':
    main()