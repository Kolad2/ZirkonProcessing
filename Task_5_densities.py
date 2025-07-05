from contextlib import nullcontext
from pathlib import Path

import numpy as np
import cv2
import scipy as sp
import json
import matplotlib.pyplot as plt


def main():
    with open("./data/zirkons_areas.json") as file:
        data = json.load(file)
    out_data_density = {}
    for name in data:
        print(name + " " + "start")
        data_density = get_density_data(data[name]["areas"])
        out_data_density[name] = data_density
    
    with open("./data/zirkons_densities.json", 'w+') as json_file:
        json.dump(out_data_density, json_file, indent=4)


def get_density_data(s):
    # вычисление размера пикселя
    s = np.delete(s, np.argmax(s))
    xmin = np.min(s)
    xmax = np.max(s)

    # Начальное число бинов
    n_bins = 10
    min_bins = 7  # минимальное допустимое число бинов

    # Логарифмические бины
    hist = None
    bins = None
    while True:
        bins = np.logspace(np.log10(xmin), np.log10(xmax), n_bins)
        hist, bins = np.histogram(s, bins=bins)
        if np.all(hist > 0):
            break
        elif n_bins > min_bins:
            n_bins -= 1
        else:
            break

    # маска для ненулевых значений гистограммы
    mask = hist > 0

    # Вычисляем плотность
    bin_widths = np.diff(bins)
    rho = np.log10(hist[mask]) - np.log10(bin_widths[mask] * np.sum(s))

    # Средние точки бинов
    s_rho = (bins[:-1] + bins[1:]) / 2
    s_rho = np.log10(s_rho[mask])

    # Преобразуем в список для JSON-сериализации
    data = {
        "s": s_rho.tolist(),
        "rho": rho.tolist(),
        "unit": "log um2"
    }
    return data


if __name__ == '__main__':
    main()
    