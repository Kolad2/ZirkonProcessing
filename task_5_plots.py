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
from plot_tools import plot_data

def main():
    font_path = Path(".") / "assets" / "timesnewromanpsmt.ttf"
    custom_font = mpl.font_manager.FontProperties(fname=font_path, size=16)

    with open("./data/zirkons_tests.json") as file:
        full_data = json.load(file)

    image_path = Path(".") / "data" / "ZirkonUpscaleSegmentation"
    raw_image_folder = Path(".") / "data" / "raw_images"

    paths = {path.name[0:path.name.find("_")-1]: path for path in image_path.iterdir()}
    raw_image_paths = {path.name[0:path.name.find(".") - 1]: path for path in raw_image_folder.iterdir()}


    for name, data in full_data.items():
        print(name)
        print(paths[name])
        print(raw_image_paths[name])
        image_segments = cv2.imread(str(paths[name]))
        image_segments = cv2.cvtColor(image_segments, cv2.COLOR_BGR2RGB)

        image_raw = cv2.imread(str(raw_image_paths[name]))
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

        fig = plt.figure(figsize=(14, 9))
        axs = [fig.add_subplot(1, 2, 1),
               fig.add_subplot(1, 2, 2)]
        axs[0].imshow(image_raw)
        axs[1].imshow(image_segments)
        plt.show()

        plot_data(data)
        exit()


if __name__ == '__main__':
    main()