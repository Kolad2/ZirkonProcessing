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
        "ecdf": {"values": values.tolist(), "freqs": e_freq.tolist()},
        "theta": {name: tests[name].theta for name, test in tests.items()}
    }
    return data


if __name__ == '__main__':
    main()