# ubuntu
# pip install opencv-contrib-python-headless
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install scipy
# python -m pip install -U scikit-image





import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyrcf


import argparse
import os
import time
import models
# from utils import *
# from edge_dataloader import BSDS_VOCLoader, BSDS_Loader, Multicue_Loader, NYUD_Loader
# from torch.utils.data import DataLoader
#
import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn


import pidinet.models as pdm
from pidinet.models.convert_pidinet import convert_pidinet
import pidinet.utils as pdu

parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')

parser.add_argument('--savedir', type=str, default='results/savedir',
        help='path to save result and checkpoint')
parser.add_argument('--datadir', type=str, default='../data',
        help='dir to the dataset')
parser.add_argument('--only-bsds', action='store_true',
        help='only use bsds for training')
parser.add_argument('--ablation', action='store_true',
        help='not use bsds val set for training')
parser.add_argument('--dataset', type=str, default='BSDS',
        help='data settings for BSDS, Multicue and NYUD datasets')

parser.add_argument('--model', type=str, default='pidinet',
        help='model to train the dataset')
parser.add_argument('--sa', action='store_true',
        help='use CSAM in pidinet')
parser.add_argument('--dil', action='store_true',
        help='use CDCM in pidinet')
parser.add_argument('--config', type=str, default='carv4',
        help='model configurations, please refer to models/config.py for possible configurations')
parser.add_argument('--seed', type=int, default=None,
        help='random seed (default: None)')
parser.add_argument('--gpu', type=str, default='',
        help='gpus available')
parser.add_argument('--checkinfo', action='store_true',
        help='only check the informations about the model: model size, flops')

parser.add_argument('--epochs', type=int, default=20,
        help='number of total epochs to run')
parser.add_argument('--iter-size', type=int, default=24,
        help='number of samples in each iteration')
parser.add_argument('--lr', type=float, default=0.005,
        help='initial learning rate for all weights')
parser.add_argument('--lr-type', type=str, default='multistep',
        help='learning rate strategy [cosine, multistep]')
parser.add_argument('--lr-steps', type=str, default=None,
        help='steps for multistep learning rate')
parser.add_argument('--opt', type=str, default='adam',
        help='optimizer')
parser.add_argument('--wd', type=float, default=1e-4,
        help='weight decay for all weights')
parser.add_argument('-j', '--workers', type=int, default=4,
        help='number of data loading workers')
parser.add_argument('--eta', type=float, default=0.3,
        help='threshold to determine the ground truth (the eta parameter in the paper)')
parser.add_argument('--lmbda', type=float, default=1.1,
        help='weight on negative pixels (the beta parameter in the paper)')

parser.add_argument('--resume', action='store_true',
        help='use latest checkpoint if have any')
parser.add_argument('--print-freq', type=int, default=10,
        help='print frequency')
parser.add_argument('--save-freq', type=int, default=1,
        help='save frequency')
parser.add_argument('--evaluate', type=str, default=None,
        help='full path to checkpoint to be evaluated')
parser.add_argument('--evaluate-converted', action='store_true',
        help='convert the checkpoint to vanilla cnn, then evaluate')
args = parser.parse_args()


DataPath = "/media/kolad/HardDisk/Zirkon/"
FileName = "Z-5c.jpg"
PathImg = DataPath + FileName
img = cv2.imread(PathImg)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# if args.seed is None:
#     args.seed = int(time.time())


if args.seed is None:
        args.seed = int(time.time())


model = pdm.pidinet(args)
#model = getattr(pdm, args.model)(args)
checkpoint = torch.load("pidinet/trained_models/table7_pidinet.pth", map_location='cpu')


def image_cv2nn(img):
        img = np.float32(img)
        img = data_loader.prepare_image_cv2(img)
        img = torch.unsqueeze(torch.from_numpy(img).cuda(), 0)
        return img

img = image_cv2nn(img)

model.eval()
with torch.no_grad():
        model(img)


#print(checkpoint)
#model.load_state_dict(checkpoint['state_dict'])



#model.load_state_dict(convert_pidinet(checkpoint['state_dict'], args.config))

exit()
#model.load_state_dict(checkpoint)
#print(checkpoint)


#


# print(convert_pidinet(checkpoint['state_dict'], args.config))

exit()

if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
else:
        raise ValueError('no checkpoint loaded')

exit()


DataPath = "/media/kolad/HardDisk/Zirkon/"
FileName = "Z-5c.jpg"
PathImg = DataPath + FileName

img = cv2.imread(PathImg)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgR, imgG, imgB = cv2.split(img)

ret, imgRB = cv2.threshold(imgR, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, imgGB = cv2.threshold(imgG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig = plt.figure(figsize=(14, 9))
fig.suptitle(FileName, fontsize=16)
axs = [fig.add_subplot(3, 2, 1),
       fig.add_subplot(3, 2, 2),
       fig.add_subplot(3, 2, 3),
       fig.add_subplot(3, 2, 4),
       fig.add_subplot(3, 2, 5),fig.add_subplot(3, 2, 6)]

axs[0].imshow(img)
axs[1].imshow(cv2.merge((imgR, imgR, imgR)))
axs[2].imshow(cv2.merge((imgG, imgG, imgG)))
axs[3].imshow(cv2.merge((imgB, imgB, imgB)))
axs[4].imshow(cv2.merge((imgRB, imgRB, imgRB)))
axs[5].imshow(cv2.merge((imgGB, imgGB, imgGB)))

for ax in axs:
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)


plt.show()
