import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def main() -> int:
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = RealESRGAN(device, scale=8)
    model.load_weights('weights/RealESRGAN_x8.pth', download=False)
    for i, pimage in enumerate(os.listdir("inputs")):
        print(pimage)
        image = Image.open(f"./inputs/{pimage}").convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(f"./results/{pimage}_up8.tiff")


if __name__ == '__main__':
    main()