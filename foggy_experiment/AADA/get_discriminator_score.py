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

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=12, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
#parser.add_argument('--asize', type=tuple, default=(376, 1244), help='size of the data (squared assumed)')
#parser.add_argument('--bsize', type=tuple, default=(1024, 2048), help='size of the data (squared assumed)'
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


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