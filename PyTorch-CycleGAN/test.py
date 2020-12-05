#!/home/vikash/anaconda3/envs/vikash/bin/python

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image
from models import Generator
from datasets_test import ImageDataset
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../foggy_experiment/data/CityScapes/leftImg8bit/original_dataset/foggyval_main', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--asize', type=tuple, default=(1024//2, 2048//2), help='size of the data (squared assumed)')
parser.add_argument('--bsize', type=tuple, default=(1024, 2048), help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='foggyC_C_model/netG_A2B.pth', help='A2B generator checkpoint file')
parser.add_argument('--generator_B2A', type=str, default='foggyC_C_model/netG_B2A.pth', help='B2A generator checkpoint file')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_A2B = torch.nn.DataParallel(netG_A2B) #uncomment if training was done in parallel
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netG_B2A = torch.nn.DataParallel(netG_B2A) # uncomment if training was done in parallel

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# ########## using data parallel model #####
# # original saved file with DataParallel
# state_dict = torch.load(opt.generator_A2B)
# # create new OrderedDict that does not contain `module.`
# from collections import OrderedDict
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# # load params
# netG_A2B.load_state_dict(new_state_dict)

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
#netG_B2A.load_state_dict(torch.load(opt.generator_B2A))


# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.asize[0], opt.asize[1])
#input_B = Tensor(opt.batchSize, opt.output_nc, opt.bsize[0], opt.bsize[1])

# Dataset loader
output_transform = transforms.Compose([transforms.Resize(opt.bsize, Image.BICUBIC),transforms.ToTensor()])
transforms_ = [ transforms.Resize(opt.asize, Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

base_path = opt.dataroot
dirs = os.listdir(base_path)
for _dir in dirs:
    dataloader = DataLoader(ImageDataset(base_path, transforms_=transforms_, mode=_dir), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
##################################

###### Testing######
    if(len(dataloader)):
        # Create output dirs if they don't exist
        output_path = os.path.join(os.path.join(base_path,'syn'),_dir)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            #print("Created directory=:datasets/CityToSyn/output/A")
        # if not os.path.exists('datasets/CityToSyn/output/city_train'):
        #     os.makedirs('datasets/CityToSyn/output/city_train')


        for i, batch in enumerate(dataloader):
            #print(i,batch['A'].size(),batch['filename'], print(type(batch['filename'])))
            #break
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            #real_B = Variable(input_B.copy_(batch['B']))

            # Generate output
            fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
            #print(fake_B)
            #fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
            #fake_B = output_transform((transforms.ToPILImage()(fake_B)))
            fake_B = F.interpolate(fake_B, size=opt.bsize)
            #print(fake_B.size())
            # Save image files
            #save_image(fake_A, 'output/A/%04d.png' % (i+1))
            output_file_path = os.path.join(output_path,batch['filename'][0])
            save_image(fake_B, output_file_path)
            sys.stdout.write('\rGenerated images =:%s' %(output_file_path))
            sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

        sys.stdout.write('\n')
    ###################################
