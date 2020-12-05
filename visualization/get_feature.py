from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

vgg = models.vgg16(pretrained=True)
vgg.classifier[6] = nn.Linear(4096,2048)
input_size = 224
batch_size = 50
data_dir = './'

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'city_syn': transforms.Compose([
        #transforms.RandomResizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'kitti_syn': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['city_syn', 'kitti_syn']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in ['city_syn', 'kitti_syn']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg = vgg.to(device)
vgg.eval()
city_syn = np.zeros((500,2048))
i = 0
for x,y in dataloaders_dict['city_syn']:
  x = x.to(device)
  out = vgg(x)
  out = out.detach().cpu().numpy()
  city_syn[i*50:(i+1)*50] = out
  i+=1
  print(out.shape)
np.save("city_syn2.npy",city_syn)

kitti_syn = np.zeros((500,2048))
i = 0
for x,y in dataloaders_dict['kitti_syn']:
  x = x.to(device)
  out = vgg(x)
  out = out.detach().cpu().numpy()
  kitti_syn[i*50:(i+1)*50] = out
  i+=1
  print(out.shape)
np.save("kitti_syn2.npy",kitti_syn)
