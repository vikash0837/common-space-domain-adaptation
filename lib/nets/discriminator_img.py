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

	def __init__(self, num_classes, ndf = 100):
		super(FCDiscriminator_img, self).__init__()

		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, padding=1)
		#self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, padding=1)

		# self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.inputsize = torch.flatten(num_classes).shape[0]
		self.fc1 = nn.Linear(100*31*62,1000)
		self.bn1 = nn.BatchNorm1d(100)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		self.fc2 = nn.Linear(1000,100)
		self.fc3 = nn.Linear(100,1)
		self.classifier = nn.Sigmoid()


	def forward(self, x):
		#print("input shape:",x.shape)
		# temp = torch.zeros((512*31*62))
		# x = torch.flatten(x)
		# temp[0:x.shape[0]] = x
		# x = temp
		# x = x.to('cuda')
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		#print("conv3 out shape:=",x.shape)

		x = self.leaky_relu(x)
		# print("input to classifier descriminator",x.shape)
		# x = self.classifier(x)
		# print("x.shape inside descriminator",x.shape)
		temp = torch.zeros((100*31*62))
		x = torch.flatten(x)
		temp[0:x.shape[0]] = x
		x = temp
		x = x.to('cuda')
		x = self.fc1(x)
		#x = x.unsqueeze(0)
		#x = self.bn1(x)
		x = self.leaky_relu(x)
		x = self.fc2(x)
		#x = x.unsqueeze(0)
		#x = self.bn2(x)
		x = self.leaky_relu(x)
		x = self.fc3(x)
		output = self.classifier(x)

		return output