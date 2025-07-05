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
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as path_effects


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
        #print(full_data[name]["theta"])
        print(full_data[name]["test_data"]["lognorm"]["ks_test"])
        
        image_segments = cv2.imread(str(paths[name]))
        image_segments = cv2.cvtColor(image_segments, cv2.COLOR_BGR2RGB)

        image_raw = cv2.imread(str(raw_image_paths[name]))
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        
        def create_scalebar(ax):
            scalebar = AnchoredSizeBar(
                ax.transData,
                size=100,  # Длина линейки в единицах данных
                label='20 units',
                loc='upper right',
                frameon=False,
                color='white',
                size_vertical=4  # Толщина линейки
            )
            text = scalebar.txt_label  # Получаем текстовый объект
            text.set_path_effects([
                path_effects.Stroke(linewidth=6, foreground='white'),  # Белая обводка
                path_effects.Normal()  # Основной текст
            ])
            scalebar.txt_label = text
            return scalebar
        
        fig = plt.figure(figsize=(14, 9))
        axs = [fig.add_subplot(1, 2, 1),
               fig.add_subplot(1, 2, 2)]
        
        axs[0].imshow(image_raw)
        axs[1].imshow(image_segments)
        axs[0].add_artist(create_scalebar(axs[0]))
        plt.show()
        
        plot_data(data)
        


if __name__ == '__main__':
    main()