import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset

import cv2

netD_A = Discriminator(3)
netD_A = torch.nn.DataParallel(netD_A)
#print("architecture:",netD_A)
netD_A.load_state_dict(torch.load("output_F2S/netD_A.pth"))
#print("with weight",netD_A)
img_path = "aachen_000000_000019_leftImg8bit_foggy_beta_0.02.png"
img = Image.open(img_path)
#trans = transforms.ToPILImage()
trans1 = transforms.ToTensor()
img = trans1(img)
if torch.cuda.is_available():
    img = img.cuda()
img = torch.unsqueeze(img, 0)
print(img.size())
print("score")
print(netD_A(img))



