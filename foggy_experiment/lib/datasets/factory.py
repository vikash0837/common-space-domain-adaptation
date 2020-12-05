# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.KITTI import KITTI
from datasets.cityscapes import cityscapes
from datasets.bdd100k import bdd100k

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}_diff'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, use_diff=True))

# Set up KITTI
for split in ['train']:#, 'val', 'synthCity', 'trainval']:
  name = 'KITTI_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: KITTI(split))
  #print("sets name=:",__sets[name])

# Set up cityscapes
for split in ['train', 'val', 'foggytrain', 'foggyval', 'synthFoggytrain', 'synthBDDdaytrain', 'synthBDDdayval']:
  name = 'cityscapes_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: cityscapes(split))

# Set up bdd100k
for split in ['train', 'val', 'daytrain', 'dayval', 'nighttrain', 'nightval', 'citydaytrain', 'citydayval', 'cleardaytrain', 'cleardayval', 'rainydaytrain', 'rainydayval']:
  name = 'bdd100k_{}'.format(split)
  __sets[name] = (lambda split=split, year=year: bdd100k(split))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  #print("sets name=",__sets)
  #print("name=",name)
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
