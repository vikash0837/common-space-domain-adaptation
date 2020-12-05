# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from model.config import cfg

# class FCDiscriminator_img(nn.Module):

# 	def __init__(self, num_classes, ndf = 64):
# 		super(FCDiscriminator_img, self).__init__()

# 		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, padding=1)
# 		self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
# 		self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
# 		self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)

# 		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


# 	def forward(self, x):
# 		x = self.conv1(x)
# 		x = self.leaky_relu(x)
# 		x = self.conv2(x)
# 		x = self.leaky_relu(x)
# 		x = self.conv3(x)
# 		x = self.leaky_relu(x)
# 		x = self.classifier(x)

# 		return x

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import cfg

class FCDiscriminator_img(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator_img, self).__init__()
		self.main = nn.Sequential(
        	# input is (nc) x 64 x 64 # 31*62
			nn.Conv2d(512, ndf, 3, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
           	# state size. (ndf) x 32 x 32 #15*30
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16 #7*14
			nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
			nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
			nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
			)

	def forward(self, input):
		#input = torch.squeeze(input)
		print("input shape:",input.shape)
		out_tensor = torch.zeros(1, 512, 31, 62)
		out_tensor[:,:,0:input.shape[2],0:input.shape[3]] = input
		out_tensor = out_tensor.to('cuda')
		result = self.main(out_tensor)
		print("result",result)
		return result