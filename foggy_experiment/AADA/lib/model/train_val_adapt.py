# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen and Zheqi He
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorboardX as tb

from model.config import cfg
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
import utils.timer
try:
  import cPickle as pickle
except ImportError:
  import pickle

import torch
import torch.optim as optim

import numpy as np
import os
import sys
import glob
import time

import json

def scale_lr(optimizer, scale):
  """Scale the learning rate of the optimizer"""
  for param_group in optimizer.param_groups:
    param_group['lr'] *= scale

class SolverWrapper(object):
  """
    A wrapper class for the training process
  """

  def __init__(self, network, imdb, roidb, imdb_T, roidb_T, valroidb, output_dir, tbdir, pretrained_model=None):
    self.net = network
    self.imdb = imdb
    self.roidb = roidb
    self.imdb_T = imdb_T
    self.roidb_T = roidb_T
    self.valroidb = valroidb
    self.output_dir = output_dir
    self.tbdir = tbdir
    # Simply put '_val' at the end to save the summaries from the validation set
    self.tbvaldir = tbdir + '_val'
    if not os.path.exists(self.tbvaldir):
      os.makedirs(self.tbvaldir)
    self.pretrained_model = pretrained_model

  def snapshot(self, iter):
    net = self.net

    if not os.path.exists(self.output_dir):
      os.makedirs(self.output_dir)

    # Store the model snapshot
    filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pth'
    filename = os.path.join(self.output_dir, filename)
    torch.save(self.net.state_dict(), filename)
    print('Wrote snapshot to: {:s}'.format(filename))

    # Also store some meta information, random state, etc.
    nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
    nfilename = os.path.join(self.output_dir, nfilename)
    # current state of numpy random
    st0 = np.random.get_state()
    # current position in the database
    cur = self.data_layer._cur
    # current shuffled indexes of the database
    perm = self.data_layer._perm
    # current position in the validation database
    cur_val = self.data_layer_val._cur
    # current shuffled indexes of the validation database
    perm_val = self.data_layer_val._perm
    # current position in the database
    curT = self.data_layer_T._cur
    # current shuffled indexes of the database
    permT = self.data_layer_T._perm

    # Dump the meta info
    with open(nfilename, 'wb') as fid:
      pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(curT, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(permT, fid, pickle.HIGHEST_PROTOCOL)
      pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

    return filename, nfilename

  def from_snapshot(self, sfile, nfile):
    print('Restoring model snapshots from {:s}'.format(sfile))
    self.net.load_state_dict(torch.load(str(sfile)))
    print('Restored.')
    # Needs to restore the other hyper-parameters/states for training, (TODO xinlei) I have
    # tried my best to find the random states so that it can be recovered exactly
    # However the Tensorflow state is currently not available
    with open(nfile, 'rb') as fid:
      st0 = pickle.load(fid)
      cur = pickle.load(fid)
      perm = pickle.load(fid)
      cur_val = pickle.load(fid)
      perm_val = pickle.load(fid)
      curT = pickle.load(fid)
      permT = pickle.load(fid)
      last_snapshot_iter = pickle.load(fid)

      np.random.set_state(st0)
      self.data_layer._cur = cur
      self.data_layer._perm = perm
      self.data_layer_val._cur = cur_val
      self.data_layer_val._perm = perm_val
      self.data_layer_T._cur = curT
      self.data_layer_T._perm = permT

    return last_snapshot_iter

  def construct_graph(self):
    # Set the random seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    # Build the main computation graph
    self.net.create_architecture(self.imdb.num_classes, tag='default',
                                            anchor_scales=cfg.ANCHOR_SCALES,
                                            anchor_ratios=cfg.ANCHOR_RATIOS)
    # Define the loss
    # loss = layers['total_loss']
    # Set learning rate and momentum
    lr = cfg.TRAIN.LEARNING_RATE
    params = []

    for key, value in dict(self.net.named_parameters()).items():
      if 'D_img' in key:
        # print(key)
        continue

      if value.requires_grad:
        # print(key)
        if 'bias' in key:
          params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
        else:
          params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    self.optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    self.D_img_op = torch.optim.SGD(self.net.D_img.parameters(), lr=lr*cfg.D_lr_mult, momentum=cfg.TRAIN.MOMENTUM)

    # Write the train and validation information to tensorboard
    self.writer = tb.writer.FileWriter(self.tbdir)
    self.valwriter = tb.writer.FileWriter(self.tbvaldir)

    return lr, self.optimizer

  def find_previous(self):
    sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pth')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in pytorch
    redfiles = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      redfiles.append(os.path.join(self.output_dir, 
                      cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.pth'.format(stepsize+1)))
    sfiles = [ss for ss in sfiles if ss not in redfiles]

    nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)
    redfiles = [redfile.replace('.pth', '.pkl') for redfile in redfiles]
    nfiles = [nn for nn in nfiles if nn not in redfiles]

    lsf = len(sfiles)
    assert len(nfiles) == lsf

    return lsf, nfiles, sfiles

  def initialize(self):
    # Initial file lists are empty
    np_paths = []
    ss_paths = []
    # Fresh train directly from ImageNet weights
    print('Loading initial model weights from {:s}'.format(self.pretrained_model))
    self.net.load_pretrained_cnn(torch.load(self.pretrained_model))
    print('Loaded.')
    # Need to fix the variables before loading, so that the RGB weights are changed to BGR
    # For VGG16 it also changes the convolutional weights fc6 and fc7 to
    # fully connected weights
    last_snapshot_iter = 0
    lr = cfg.TRAIN.LEARNING_RATE
    stepsizes = list(cfg.TRAIN.STEPSIZE)

    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def restore(self, sfile, nfile):
    # Get the most recent snapshot and restore
    np_paths = [nfile]
    ss_paths = [sfile]
    # Restore model from snapshots
    last_snapshot_iter = self.from_snapshot(sfile, nfile)
    # Set the learning rate
    lr_scale = 1
    stepsizes = []
    for stepsize in cfg.TRAIN.STEPSIZE:
      if last_snapshot_iter > stepsize:
        lr_scale *= cfg.TRAIN.GAMMA
      else:
        stepsizes.append(stepsize)
    scale_lr(self.optimizer, lr_scale)
    lr = cfg.TRAIN.LEARNING_RATE * lr_scale
    return lr, last_snapshot_iter, stepsizes, np_paths, ss_paths

  def remove_snapshot(self, np_paths, ss_paths):
    to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      nfile = np_paths[0]
      os.remove(str(nfile))
      np_paths.remove(nfile)

    to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
    for c in range(to_remove):
      sfile = ss_paths[0]
      # To make the code compatible to earlier versions of Tensorflow,
      # where the naming tradition for checkpoints are different
      os.remove(str(sfile))
      ss_paths.remove(sfile)
    
  def train_model(self, max_iters):
    MIN_TOTAT_LOSS = np.inf
    MIN_D_LOSS_T = np.inf
    BEST_ITER = None
    # Build data layers for both training and validation set
    self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
    self.data_layer_val = RoIDataLayer(self.valroidb, self.imdb.num_classes, random=True)
    self.data_layer_T = RoIDataLayer(self.roidb_T, self.imdb.num_classes)

    # Construct the computation graph
    lr, train_op = self.construct_graph()

    # Find previous snapshots if there is any to restore from
    lsf, nfiles, sfiles = self.find_previous()

    # Initialize the variables or restore them from the last snapshot
    if lsf == 0:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.initialize()
    else:
      lr, last_snapshot_iter, stepsizes, np_paths, ss_paths = self.restore(str(sfiles[-1]), 
                                                                             str(nfiles[-1]))
    iter = last_snapshot_iter + 1
    last_summary_time = time.time()
    # Make sure the lists are not empty
    stepsizes.append(max_iters)
    stepsizes.reverse()
    next_stepsize = stepsizes.pop()

    self.net.train()
    self.net.cuda()

    self.net.D_img.train()
    self.net.D_img.cuda()

    #self.net.D_img2.train()
    #self.net.D_img2.cuda()
    mywriter = tb.SummaryWriter()
    while iter < max_iters + 1:
      # Learning rate
      if iter == next_stepsize + 1:
        # Add snapshot here before reducing the learning rate
        self.snapshot(iter)
        lr *= cfg.TRAIN.GAMMA
        scale_lr(self.optimizer, cfg.TRAIN.GAMMA)
        #scale_lr(self.D_img_op, cfg.TRAIN.GAMMA)
        next_stepsize = stepsizes.pop()

      utils.timer.timer.tic()
      # Get training data, one batch at a time
      blobs = self.data_layer.forward()
      blobsT = self.data_layer_T.forward()
      # print("#########################:blobs['data_path'][0]",blobs['data_path'][0])
      # print("synth_weight:",imdb.D_T_score[os.path.basename(blobs['data_path'][0])])
      # break
      now = time.time()
      #if iter == 1 or now - last_summary_time > cfg.TRAIN.SUMMARY_INTERVAL:
      if False: 
        # Compute the graph with summary
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, D_inst_loss_S, D_img_loss_S, D_const_loss_S, D_inst_loss_T, D_img_loss_T, D_const_loss_T, summary = \
          self.net.train_adapt_step_with_summary(blobs, blobsT, self.optimizer, self.D_inst_op, self.D_img_op)
        for _sum in summary: self.writer.add_summary(_sum, float(iter))
        # Also check the summary on the validation set
        blobs_val = self.data_layer_val.forward()
        summary_val = self.net.get_summary(blobs_val)
        for _sum in summary_val: self.valwriter.add_summary(_sum, float(iter))
        last_summary_time = now
      else:
        # Compute the graph without summary
        print("##########blobs['data_path'][0]#######:",blobs['data_path'][0])
        if 'train' in blobs['data_path'][0]:
          synth_weight = self.imdb.D_T_score[os.path.basename(blobs['data_path'][0])]
          print("Synth weight is=:",synth_weight)
        else:
          synth_weight = 1
        
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, total_loss, D_img_loss_S, D_img_loss_T = \
            self.net.train_adapt_step_img(blobs, blobsT, self.optimizer, self.D_img_op, synth_weight)

      utils.timer.timer.toc()
      if(((loss_cls+loss_box) < MIN_TOTAT_LOSS) and (D_img_loss_T < MIN_D_LOSS_T)):
        MIN_TOTAT_LOSS = loss_cls+loss_box
        MIN_D_LOSS_T = D_img_loss_T
        BEST_ITER = iter
        print("Curr MIN_TOTAT_LOSS=:{} and min_D_loss_T=:{}".format(MIN_TOTAT_LOSS,MIN_D_LOSS_T))

      # Display training information
      if iter % (cfg.TRAIN.DISPLAY) == 0:
        mywriter.add_scalar("Total Loss",total_loss,iter)
        mywriter.add_scalar("cls Loss",loss_cls,iter)
        mywriter.add_scalar("Box loss",loss_box,iter)
        mywriter.add_scalar("D_img_loss_S",D_img_loss_S,iter)
        mywriter.add_scalar("D_img_loss_T",D_img_loss_T,iter)
        fp = open('training_log.txt','a+')
        print("Writing Training log")
        temp_log = str(iter) + ',' + str(total_loss) + ',' + str(rpn_loss_cls) + ',' + str(rpn_loss_box) + ',' + str(loss_cls)+',' + str(loss_box) + ',' + str(D_img_loss_S) + ',' + str(D_img_loss_T)
        fp.write(temp_log)
        fp.write('\n')
        fp.close()
        print("Done !!!!!!!!")
        print('iter: %d / %d, total loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
              '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n '
              '>>> D_img_loss_S: %.6f\n >>> D_img_loss_T: %.6f\n '
              '>>> lambda: %f >>> lr: %f ' % \
              (iter, max_iters, total_loss, rpn_loss_cls, \
                rpn_loss_box, loss_cls, loss_box, \
                D_img_loss_S, D_img_loss_T, \
                cfg.ADAPT_LAMBDA, lr))
        print('speed: {:.3f}s / iter'.format(utils.timer.timer.average_time()))

        # for k in utils.timer.timer._average_time.keys():
        #   print(k, utils.timer.timer.average_time(k))

      # Snapshotting
      if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
        last_snapshot_iter = iter
        ss_path, np_path = self.snapshot(iter)
        np_paths.append(np_path)
        ss_paths.append(ss_path)

        # Remove the old snapshots if there are too many
        if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
          self.remove_snapshot(np_paths, ss_paths)

      iter += 1
    print("##################### Best iteration = ",BEST_ITER,"###################################")
    print("##################### Optimal Total Loss=:",MIN_TOTAT_LOSS,"###########################")
    print("##################### Optimal lD_LOSS_T=:",MIN_D_LOSS_T,"##############################")
    _,_ = self.snapshot(BEST_ITER)
    if last_snapshot_iter != iter - 1:
      self.snapshot(iter - 1)

    self.writer.close()
    self.valwriter.close()


def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def filter_roidb(roidb):
  """Remove roidb entries that have no usable RoIs."""

  def is_valid(entry):
    # Valid images have:
    #   (1) At least one foreground RoI OR
    #   (2) At least one background RoI
    overlaps = entry['max_overlaps']
    # find boxes with sufficient overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # image is only valid if such boxes exist
    valid = len(fg_inds) > 0 or len(bg_inds) > 0

    return valid

  num = len(roidb)
  filtered_roidb = [entry for entry in roidb if is_valid(entry)]
  num_after = len(filtered_roidb)
  print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                     num, num_after))
  return filtered_roidb


def train_net(network, imdb, roidb, imdb_T, roidb_T, valroidb, output_dir, tb_dir,
              pretrained_model=None,
              max_iters=40000):
  """Train a Faster R-CNN network."""
  roidb = filter_roidb(roidb)
  valroidb = filter_roidb(valroidb)

  roidb_T = filter_roidb(roidb_T)

  sw = SolverWrapper(network, imdb, roidb, imdb_T, roidb_T, valroidb, output_dir, tb_dir,
                     pretrained_model=pretrained_model)

  print('Solving...')
  sw.train_model(max_iters)
  print('done solving')
