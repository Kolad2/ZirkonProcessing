import math
import os
import numpy as np
import cv2
from scipy.io import savemat
from pyrocksegmentation.basic_segmentator import Segmentator
from pyrocksegmentation import Extractor
import matplotlib.pyplot as plt
from scipy import stats as st
from pathlib import Path
from tqdm import tqdm
import json


def main():
    # Path_dir_edges = "/media/kolad/HardDisk/Zirkon/ZirkonUpscaleBINEdgesPrep"
    # Path_dir_imgs = "/media/kolad/HardDisk/Zirkon/ZirkonUpscale"
    # Path_dir_segs = "/media/kolad/HardDisk/Zirkon/ZirkonUpscaleSegmentation"
    
    path_dir_edges = Path(".\data\ZirkonEdges")
    
    file_names = os.listdir(str(path_dir_edges))
    
    with open("./data/config.json") as file:
        config = json.load(file)
    
    data = {}
    for file_name in tqdm(file_names):
        name = file_name[0:file_name.find('_')-1]
        path_img_edges = path_dir_edges / file_name
        image_edges = cv2.imread(str(path_img_edges))
        areas = zirkon_image_to_areas(image_edges)
        areas = areas*(config[name]["um"]/config[name]["pix"])
        areas = [round(num, 2) for num in areas.tolist()]
        areas_data = {"areas": areas, "units": "um"}
        data[name] = areas_data
    
    with open("data/zirkons_areas.json", 'w+') as json_file:
        json.dump(data, json_file, indent=4)


def zirkon_image_to_areas(image):
    b, g, r = cv2.split(image)
    edges = np.zeros(b.shape)
    edges[(b == 0) & (r == 255)] = 255
    
    segments = Segmentator(g).run()
    segments[edges == 255] = -1
    areas, centers = Extractor(segments).extruct_centers(indent=1)
    return areas


if __name__ == '__main__':
    main()
