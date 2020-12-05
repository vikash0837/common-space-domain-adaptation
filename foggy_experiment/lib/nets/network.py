# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.transforms import Normalize, ToTensor, Resize

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer, proposal_layer_fpn
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from layer_utils.roi_pooling.roi_pool import RoIPoolFunction
from layer_utils.roi_align.crop_and_resize import CropAndResizeFunction

from model.config import cfg

import tensorboardX as tb

from scipy.misc import imresize

from nets.discriminator_img import FCDiscriminator_img
import cv2
from PIL import Image
from model.bbox_transform import bbox_transform_inv
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.mm(input2_l2.t()).pow(2)))

        return diff_loss

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def printgradnorm(self, grad_input, grad_output):
    #print('Inside ' + self.__class__.__name__ + ' backward')
    #print('Inside class:' + self.__class__.__name__)
    #print('')
    #print('grad_input: ', type(grad_input))
    #print('grad_input[0]: ', type(grad_input[0]))
    #print('grad_output: ', type(grad_output))
    #print('grad_output[0]: ', type(grad_output[0]))
    #print('')
    #print('grad_input size:', grad_input[0].size())
    #print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].data.norm())

class Network(nn.Module):
  def __init__(self):
    nn.Module.__init__(self)
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._act_summaries = {}
    self._score_summaries = {}
    self._event_summaries = {}
    self._image_gt_summaries = {}
    self._variables_to_fix = {}

  def _add_gt_image(self):
    # add back mean
    image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
    image = imresize(image[0], self._im_info[:2] / self._im_info[2])
    # BGR to RGB (opencv uses BGR)
    self._gt_image = image[np.newaxis, :,:,::-1].copy(order='C')

  def _add_gt_image_summary(self):
    # use a customized visualization function to visualize the boxes
    self._add_gt_image()
    image = draw_bounding_boxes(\
                      self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])

    return tb.summary.image('GROUND_TRUTH', image[0].astype('float32')/255.0)

  def _add_act_summary(self, key, tensor):
    return tb.summary.histogram('ACT/' + key + '/activations', tensor.data.cpu().numpy(), bins='auto'),
    tb.summary.scalar('ACT/' + key + '/zero_fraction',
                      (tensor.data == 0).float().sum() / tensor.numel())

  def _add_score_summary(self, key, tensor):
    return tb.summary.histogram('SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')

  def _add_train_summary(self, key, var):
    return tb.summary.histogram('TRAIN/' + key, var.data.cpu().numpy(), bins='auto')

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     self._feat_stride, self._anchors, self._num_anchors)
    return rois, rpn_scores

  def _proposal_top_layer_fpn(self, rpn_cls_prob, rpn_bbox_pred, idx):
    rois, rpn_scores = proposal_top_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                     [self._feat_stride[idx]], self._anchors[idx], self._num_anchors)
    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_layer(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)

    return rois, rpn_scores

  def _proposal_layer_fpn(self, rpn_cls_prob, rpn_bbox_pred):
    rois, rpn_scores = proposal_layer_fpn(\
                                    rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors)

    return rois, rpn_scores

  def _roi_pool_layer(self, bottom, rois):
    return RoIPoolFunction(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1. / 16.)(bottom, rois)

  def _crop_pool_layer(self, bottom, rois, max_pool=True):
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()

    x1 = rois[:, 1::4] / 16.0
    y1 = rois[:, 2::4] / 16.0
    x2 = rois[:, 3::4] / 16.0
    y2 = rois[:, 4::4] / 16.0

    height = bottom.size(2)
    width = bottom.size(3)

    pre_pool_size = cfg.POOLING_SIZE * 2 if max_pool else cfg.POOLING_SIZE
    crops = CropAndResizeFunction(pre_pool_size, pre_pool_size)(bottom, 
      torch.cat([y1/(height-1),x1/(width-1),y2/(height-1),x2/(width-1)], 1), rois[:, 0].int())
    if max_pool:
      crops = F.max_pool2d(crops, 2, 2)
    return crops

  def _crop_pool_layer_fpn(self, bottom, rois, max_pool=True):
    # implement it using stn
    # box to affine
    # input (x1,y1,x2,y2)
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """
    rois = rois.detach()
    roi_num = rois.size(0)

    x1 = rois[:, 1::4]
    y1 = rois[:, 2::4]
    x2 = rois[:, 3::4]
    y2 = rois[:, 4::4]

    k0 = 4
    w = x2-x1
    h = y2-y1
    w[w<=0]=1e-14
    h[h<=0]=1e-14
    ratio = torch.sqrt(w*h) / 224.
    k = k0 + np.log2(ratio.cpu().data.numpy())
    k[k<2]=2
    k[k>5]=5
    k = Variable(torch.round(torch.from_numpy(k)).cuda())
    x1 = x1 / (2 ** k)
    y1 = y1 / (2 ** k)
    x2 = x2 / (2 ** k)
    y2 = y2 / (2 ** k)

    k = k.long()

    height = []
    width = []
    for idx in range(k.size(0)):
      height.append(bottom[k[idx].data[0]-2].size(2))
      width.append(bottom[k[idx].data[0]-2].size(3))

    height = Variable(torch.Tensor(height).cuda()).view(roi_num,1)
    width = Variable(torch.Tensor(width).cuda()).view(roi_num,1)

    # affine theta
    theta = Variable(rois.data.new(rois.size(0), 2, 3).zero_())
    theta[:, 0, 0] = ((x2 - x1) / (width - 1)).squeeze()
    theta[:, 0 ,2] = ((x1 + x2 - width + 1) / (width - 1)).squeeze()
    theta[:, 1, 1] = ((y2 - y1) / (height - 1)).squeeze()
    theta[:, 1, 2] = ((y1 + y2 - height + 1) / (height - 1)).squeeze()

    if max_pool:
      pre_pool_size = cfg.POOLING_SIZE * 2
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, pre_pool_size, pre_pool_size)))

      all_roi = []
      for j in range(rois.size(0)):
        _grid = grid.narrow(0, j, 1)
        _roi_feature = F.grid_sample(bottom[k[j].data[0]-2].view(1,bottom[k[j].data[0]-2].size(1), bottom[k[j].data[0]-2].size(2), bottom[k[j].data[0]-2].size(3)), _grid)
        all_roi.append(_roi_feature)
      crops = torch.cat(all_roi)
      # crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)
      crops = F.max_pool2d(crops, 2, 2)
    else:
      grid = F.affine_grid(theta, torch.Size((rois.size(0), 1, cfg.POOLING_SIZE, cfg.POOLING_SIZE)))

      all_roi = []
      for j in range(rois.size(0)):
        _grid = grid.narrow(0, j, 1)
        _roi_feature = F.grid_sample(bottom[k[j].data[0]-2].view(1,bottom[k[j].data[0]-2].size(1), bottom[k[j].data[0]-2].size(2), bottom[k[j].data[0]-2].size(3)), _grid)
        all_roi.append(_roi_feature)
      crops = torch.cat(all_roi)
      # crops = F.grid_sample(bottom.expand(rois.size(0), bottom.size(1), bottom.size(2), bottom.size(3)), grid)

    return crops

  def _anchor_target_layer(self, rpn_cls_score):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)

    rpn_labels = Variable(torch.from_numpy(rpn_labels).float().cuda()) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = Variable(torch.from_numpy(rpn_bbox_targets).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = Variable(torch.from_numpy(rpn_bbox_inside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = Variable(torch.from_numpy(rpn_bbox_outside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])

    rpn_labels = rpn_labels.long()
    self._anchor_targets['rpn_labels'] = rpn_labels
    self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
    self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
    self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

    for k in self._anchor_targets.keys():
      self._score_summaries[k] = self._anchor_targets[k]

    return rpn_labels

  def _anchor_target_layer_fpn(self, rpn_cls_score, idx):
    rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
      anchor_target_layer(
      rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._im_info, [self._feat_stride[idx]], self._anchors[idx].data.cpu().numpy(), self._num_anchors)

    rpn_labels = Variable(torch.from_numpy(rpn_labels).float().cuda()) #.set_shape([1, 1, None, None])
    rpn_bbox_targets = Variable(torch.from_numpy(rpn_bbox_targets).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_inside_weights = Variable(torch.from_numpy(rpn_bbox_inside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])
    rpn_bbox_outside_weights = Variable(torch.from_numpy(rpn_bbox_outside_weights).float().cuda())#.set_shape([1, None, None, self._num_anchors * 4])

    rpn_labels = rpn_labels.long()
    if 'rpn_labels' not in self._anchor_targets:
      self._anchor_targets['rpn_labels'] = []
    if 'rpn_bbox_targets' not in self._anchor_targets:
      self._anchor_targets['rpn_bbox_targets'] = []
    if 'rpn_bbox_inside_weights' not in self._anchor_targets:
      self._anchor_targets['rpn_bbox_inside_weights'] = []
    if 'rpn_bbox_outside_weights' not in self._anchor_targets:
      self._anchor_targets['rpn_bbox_outside_weights'] = []
    self._anchor_targets['rpn_labels'].append(rpn_labels)
    self._anchor_targets['rpn_bbox_targets'].append(rpn_bbox_targets)
    self._anchor_targets['rpn_bbox_inside_weights'].append(rpn_bbox_inside_weights)
    self._anchor_targets['rpn_bbox_outside_weights'].append(rpn_bbox_outside_weights)

    return rpn_labels

  def _proposal_target_layer(self, rois, roi_scores):
    rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
      proposal_target_layer(
      rois, roi_scores, self._gt_boxes, self._num_classes)

    self._proposal_targets['rois'] = rois
    self._proposal_targets['labels'] = labels.long()
    self._proposal_targets['bbox_targets'] = bbox_targets
    self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
    self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

    for k in self._proposal_targets.keys():
      self._score_summaries[k] = self._proposal_targets[k]

    return rois, roi_scores

  def _anchor_component(self, height, width):
    # just to get the shape right
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    anchors, anchor_length = generate_anchors_pre(\
                                          height, width,
                                           self._feat_stride, self._anchor_scales, self._anchor_ratios)
    self._anchors = Variable(torch.from_numpy(anchors).cuda())
    self._anchor_length = anchor_length

  def _anchor_component_fpn(self, net_conv):
    # just to get the shape right
    #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
    #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
    anchors_total = []
    anchor_length_total = []
    for idx, p in enumerate(net_conv):
      height = p.size(2)
      width = p.size(3) 

      anchors, anchor_length = generate_anchors_pre(\
                                            height, width,
                                             [self._feat_stride[idx]], [self._anchor_scales[idx]], self._anchor_ratios)
      anchors_total.append(Variable(torch.from_numpy(anchors).cuda()))
      anchor_length_total.append(anchor_length)

    self._anchors = anchors_total
    self._anchor_length = anchor_length_total

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()
    return loss_box

  def _add_losses(self, sigma_rpn=3.0):
    # RPN, class loss
    rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
    rpn_label = self._anchor_targets['rpn_labels'].view(-1)
    rpn_select = Variable((rpn_label.data != -1).nonzero().view(-1))
    rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)
    rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    # RPN, bbox loss
    rpn_bbox_pred = self._predictions['rpn_bbox_pred']
    rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
    rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
    rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
    rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

    # RCNN, class loss
    cls_score = self._predictions["cls_score"]
    label = self._proposal_targets["labels"].view(-1)
    cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes), label)

    # RCNN, bbox loss
    bbox_pred = self._predictions['bbox_pred']
    bbox_targets = self._proposal_targets['bbox_targets']
    bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
    bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
    loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    self._losses['cross_entropy'] = cross_entropy
    self._losses['loss_box'] = loss_box
    self._losses['rpn_cross_entropy'] = rpn_cross_entropy
    self._losses['rpn_loss_box'] = rpn_loss_box

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
    self._losses['total_loss'] = loss

    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]

    return loss

  def _add_losses_fpn(self, sigma_rpn=3.0):
    rpn_cross_entropy = 0
    rpn_cross_entropy_num = 0
    rpn_loss_box = 0
    rpn_loss_box_num = 0
    for idx in range(5):
      # RPN, class loss
      rpn_cls_score = self._predictions['rpn_cls_score_reshape'][idx].view(-1, 2)
      rpn_label = self._anchor_targets['rpn_labels'][idx].view(-1)
      rpn_select = Variable((rpn_label.data != -1).nonzero().view(-1))
      rpn_cls_score = rpn_cls_score.index_select(0, rpn_select).contiguous().view(-1, 2)
      rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
      rpn_cross_entropy += F.cross_entropy(rpn_cls_score, rpn_label) * rpn_label.size(0)
      rpn_cross_entropy_num += rpn_label.size(0)

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred'][idx]
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets'][idx]
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights'][idx]
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights'][idx]
      rpn_loss_box += self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                            rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3]) * rpn_bbox_inside_weights.sum()
      rpn_loss_box_num += rpn_bbox_inside_weights.sum()

    # TODO NORMALIZE FOR LOSS
    rpn_cross_entropy /= rpn_cross_entropy_num
    rpn_loss_box /= rpn_loss_box_num

    # RCNN, class loss
    cls_score = self._predictions["cls_score"]
    label = self._proposal_targets["labels"].view(-1)
    cross_entropy = F.cross_entropy(cls_score.view(-1, self._num_classes), label)

    # RCNN, bbox loss
    bbox_pred = self._predictions['bbox_pred']
    bbox_targets = self._proposal_targets['bbox_targets']
    bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
    bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
    loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

    self._losses['cross_entropy'] = cross_entropy
    self._losses['loss_box'] = loss_box
    self._losses['rpn_cross_entropy'] = rpn_cross_entropy
    self._losses['rpn_loss_box'] = rpn_loss_box

    loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
    self._losses['total_loss'] = loss

    for k in self._losses.keys():
      self._event_summaries[k] = self._losses[k]

    return loss

  def _region_proposal(self, net_conv):
    rpn = F.relu(self.rpn_net(net_conv))
    self._act_summaries['rpn'] = rpn

    rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w

    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = rpn_cls_score.view(1, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w
    rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
    
    # Move channel to the last dimenstion, to fit the input of python functions
    rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
    rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
    rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

    rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
    rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

    if self._mode == 'TRAIN':
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred) # rois, roi_scores are varible ##error
      rpn_labels = self._anchor_target_layer(rpn_cls_score)
      rois, _ = self._proposal_target_layer(rois, roi_scores)
    else:
      if cfg.TEST.MODE == 'nms':
        rois, self.roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  def _region_proposal_fpn(self, net_conv):
    # self._act_summaries['rpn'] = []
    rpn_cls_prob_total = []
    rpn_bbox_pred_total = []
    for idx, p in enumerate(net_conv):
      rpn = F.relu(self.rpn_net(p))
      # self._act_summaries['rpn'].append(rpn)

      rpn_cls_score = self.rpn_cls_score_net(rpn) # batch * (num_anchors * 2) * h * w

      # change it so that the score has 2 as its channel size
      rpn_cls_score_reshape = rpn_cls_score.view(1, 2, -1, rpn_cls_score.size()[-1]) # batch * 2 * (num_anchors*h) * w
      rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape)

      # Move channel to the last dimenstion, to fit the input of python functions
      rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
      rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1) # batch * h * w * (num_anchors * 2)
      rpn_cls_score_reshape = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
      rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

      rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
      rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

      rpn_cls_prob_total.append(rpn_cls_prob)
      rpn_bbox_pred_total.append(rpn_bbox_pred)

      if self._mode == 'TRAIN':
        rpn_labels = self._anchor_target_layer_fpn(rpn_cls_score,idx)

      if 'rpn_cls_score' not in self._predictions:
        self._predictions['rpn_cls_score'] = []
      if 'rpn_cls_score_reshape' not in self._predictions:
        self._predictions['rpn_cls_score_reshape'] = []
      if 'rpn_cls_prob' not in self._predictions:
        self._predictions['rpn_cls_prob'] = []
      if 'rpn_cls_pred' not in self._predictions:
        self._predictions['rpn_cls_pred'] = []
      if 'rpn_bbox_pred' not in self._predictions:
        self._predictions['rpn_bbox_pred'] = []
      # self._predictions["rpn_cls_score"].append(rpn_cls_score)
      self._predictions["rpn_cls_score_reshape"].append(rpn_cls_score_reshape)
      # self._predictions["rpn_cls_prob"].append(rpn_cls_prob)
      # self._predictions["rpn_cls_pred"].append(rpn_cls_pred)
      self._predictions["rpn_bbox_pred"].append(rpn_bbox_pred)

    if self._mode == 'TRAIN':
      rois, roi_scores = self._proposal_layer_fpn(rpn_cls_prob_total, rpn_bbox_pred_total) # rois, roi_scores are varible
      rois, _ = self._proposal_target_layer(rois, roi_scores)
      # for k in self._anchor_targets.keys():
      #   self._score_summaries[k] = self._anchor_targets[k]
    else:
      # TODO
      if cfg.TEST.MODE == 'nms':
        rois, self.roi_scores = self._proposal_layer_fpn(rpn_cls_prob_total, rpn_bbox_pred_total)
      # elif cfg.TEST.MODE == 'top':
      #   rois, _ = self._proposal_top_layer_fpn(rpn_cls_prob, rpn_bbox_pred)
      else:
        raise NotImplementedError

    self._predictions["rois"] = rois

    return rois

  def _region_classification(self, fc7):
    cls_score = self.cls_score_net(fc7)
    cls_pred = torch.max(cls_score, 1)[1]
    cls_prob = F.softmax(cls_score, dim=1)
    bbox_pred = self.bbox_pred_net(fc7)

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _image_to_head(self):
    raise NotImplementedError

  def _head_to_tail(self, pool5):
    raise NotImplementedError

  def create_architecture(self, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    self._tag = tag

    self._num_classes = num_classes
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # Initialize layers
    self._init_modules()

  def _init_modules(self):
    self._init_head_tail()

    # rpn
    self.rpn_net = nn.Conv2d(self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)

    self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 2, [1, 1])
    
    self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 4, [1, 1])

    self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)
    self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes * 4)

    #discriminator for instance and image level
    self.D_img = FCDiscriminator_img(self._net_conv_channels)

    self.init_weights()

  def _run_summary_op(self, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():
      summaries.append(tb.summary.scalar(key, var.data[0]))
    self._event_summaries = {}
    if not val:
      # Add score summaries
      for key, var in self._score_summaries.items():
        summaries.append(self._add_score_summary(key, var))
      self._score_summaries = {}
      # Add act summaries
      for key, var in self._act_summaries.items():
        summaries += self._add_act_summary(key, var)
      self._act_summaries = {}
      # Add train summaries
      for k, var in dict(self.named_parameters()).items():
        if var.requires_grad:
          summaries.append(self._add_train_summary(k, var))

      self._image_gt_summaries = {}
    
    return summaries

  def _run_summary_op_fpn(self, val=False):
    """
    Run the summary operator: feed the placeholders with corresponding newtork outputs(activations)
    """
    summaries = []
    # Add image gt
    summaries.append(self._add_gt_image_summary())
    # Add event_summaries
    for key, var in self._event_summaries.items():
      summaries.append(tb.summary.scalar(key, var.data[0]))
    self._event_summaries = {}
    # if not val:
    #   # Add score summaries
    #   for key, var in self._score_summaries.items():
    #     if key.startswith('rpn'):
    #       continue
    #     summaries.append(self._add_score_summary(key, var))
    #   self._score_summaries = {}
    #   # Add act summaries
    #   # for key, var in self._act_summaries.items():
    #   #   summaries += self._add_act_summary(key, var)
    #   # self._act_summaries = {}
    #   # Add train summaries
    #   for k, var in dict(self.named_parameters()).items():
    #     if var.requires_grad:
    #       summaries.append(self._add_train_summary(k, var))

    self._image_gt_summaries = {}

    return summaries

  def _predict(self):
    # This is just _build_network in tf-faster-rcnn
    torch.backends.cudnn.benchmark = False
    net_conv = self._image_to_head()

    # build the anchors for the image
    self._anchor_component(net_conv.size(2), net_conv.size(3))
    rois = self._region_proposal(net_conv)

    if cfg.POOLING_MODE == 'crop':
      pool5 = self._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self._roi_pool_layer(net_conv, rois)

    if self._mode == 'TRAIN':
      torch.backends.cudnn.benchmark = True # benchmark because now the input size are fixed
    fc7 = self._head_to_tail(pool5)

    cls_prob, bbox_pred = self._region_classification(fc7)

    for k in self._predictions.keys():
      self._score_summaries[k] = self._predictions[k]

    return rois, cls_prob, bbox_pred, net_conv, fc7
    # return rois, cls_prob, bbox_pred, net_conv_orig, fc7 #feature_separate
    
  def _clip_boxes(self, boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes
  
  def forward(self, image, im_info, gt_boxes=None, mode='TRAIN', adapt=None):
    self._image_gt_summaries['image'] = image
    self._image_gt_summaries['gt_boxes'] = gt_boxes
    self._image_gt_summaries['im_info'] = im_info

    self._image = Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=mode == 'TEST')
    self._im_info = im_info # No need to change; actually it can be an list
    self._gt_boxes = Variable(torch.from_numpy(gt_boxes).cuda()) if gt_boxes is not None else None

    self._mode = mode

    rois, cls_prob, bbox_pred, net_conv, fc7 = self._predict()

    if mode == 'TEST':
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self._num_classes).unsqueeze(0).expand_as(bbox_pred)
      self._predictions["bbox_pred"] = bbox_pred.mul(Variable(stds)).add(Variable(means))
    elif adapt:
      pass
    else:
      self._add_losses() # compute losses

    return fc7, net_conv

  def init_weights(self):
    def G_weights_init_normal(m):
      classname = m.__class__.__name__
      if classname.find('Conv') != -1:
          torch.nn.init.normal(m.weight.data, 0.0, 0.02)
      elif classname.find('BatchNorm2d') != -1:
          torch.nn.init.normal(m.weight.data, 1.0, 0.02)
          torch.nn.init.constant(m.bias.data, 0.0)
    def weights_init(m):
      classname = m.__class__.__name__
      if classname.find('Conv') != -1:
          nn.init.normal_(m.weight.data, 0.0, 0.02)
      elif classname.find('BatchNorm') != -1:
          nn.init.normal_(m.weight.data, 1.0, 0.02)
          nn.init.constant_(m.bias.data, 0)
    def normal_init(m, mean, stddev, truncated=False):
      """
      weight initalizer: truncated normal and random normal.
      """
      # x is a parameter
      if truncated:
        m.weight.data.normal_(generator=torch.manual_seed(cfg.RNG_SEED)).fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
      else:
        m.weight.data.normal_(mean, stddev, generator=torch.manual_seed(cfg.RNG_SEED))
      m.bias.data.zero_()
      
    normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
    normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

    #normal_init(self.D_img, 0, 0.01, cfg.TRAIN.TRUNCATED)
    self.D_img.apply(weights_init)

    # normal_init(self.D_img.conv2, 0, 0.01, cfg.TRAIN.TRUNCATED)
    # normal_init(self.D_img.conv3, 0, 0.01, cfg.TRAIN.TRUNCATED)
    # normal_init(self.D_img.fc1, 0, 0.01, cfg.TRAIN.TRUNCATED)
    # normal_init(self.D_img.fc2, 0, 0.01, cfg.TRAIN.TRUNCATED)
    # normal_init(self.D_img.fc3, 0, 0.01, cfg.TRAIN.TRUNCATED)
    #normal_init(self.D_img.classifier, 0, 0.01, cfg.TRAIN.TRUNCATED)

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, image):
    feat = self._layers["head"](Variable(torch.from_numpy(image.transpose([0,3,1,2])).cuda(), volatile=True))
    return feat

  # only useful during testing mode
  def test_discriminator(self, image, im_info):
    self.eval()
    fc7, net_conv = self.forward(image, im_info, None, mode='TEST')
    net_conv = grad_reverse(net_conv)
    #D_img
    D_img_out = self.D_img(net_conv)
    return fc7, net_conv,D_img_out



  def test_image(self, image, im_info):
    self.eval()
    fc7, net_conv = self.forward(image, im_info, None, mode='TEST')
    cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
                                                     self._predictions['cls_prob'].data.cpu().numpy(), \
                                                     self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                     self._predictions['rois'].data.cpu().numpy()
    self.delete_intermediate_states()
    return cls_score, cls_prob, bbox_pred, rois, fc7, net_conv

  def delete_intermediate_states(self):
    # Delete intermediate result to save memory
    for d in [self._losses, self._predictions, self._anchor_targets, self._proposal_targets]:
      for k in list(d):
        del d[k]

  def get_summary(self, blobs):
    self.eval()
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    self.train()
    summary = self._run_summary_op(True)

    return summary

  def train_adapt_step_img(self, blobs_S, blobs_T, train_op, D_img_op, synth_weight):
    source_label = 0
    target_label = 1

    train_op.zero_grad()
    D_img_op.zero_grad()
    
    bceLoss_func = nn.BCEWithLogitsLoss()

    #train with source
    fc7, net_conv = self.forward(blobs_S['data'], blobs_S['im_info'], blobs_S['gt_boxes'])

    net_conv = grad_reverse(net_conv)

    #det loss
    loss_S = self._losses['total_loss'] * synth_weight

    #D_img
    D_img_out = self.D_img(net_conv)

    #loss
    loss_D_img_S = bceLoss_func(D_img_out, Variable(torch.FloatTensor(D_img_out.data.size()).fill_(source_label)).cuda())
    
    total_loss_S = loss_S + (cfg.ADAPT_LAMBDA/2.) * loss_D_img_S

    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    #train with target
    fc7, net_conv = self.forward(blobs_T['data'], blobs_T['im_info'], blobs_T['gt_boxes'], adapt=True)
    net_conv = grad_reverse(net_conv)

    #D_img
    D_img_out = self.D_img(net_conv)
    #loss
    loss_D_img_T = bceLoss_func(D_img_out, Variable(torch.FloatTensor(D_img_out.data.size()).fill_(target_label)).cuda())

    total_loss_T = (cfg.ADAPT_LAMBDA/2.) * loss_D_img_T

    total_loss = total_loss_S + total_loss_T
    total_loss.backward()
    
    #clip gradient
    # clip = 10
    # torch.nn.utils.clip_grad_norm(self.D_img.parameters(),clip)
    # torch.nn.utils.clip_grad_norm(self.parameters(),clip)

    train_op.step()
    D_img_op.step()
                                                                        
    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, loss_D_img_S, loss_D_img_T

  def FPN_train_adapt_step_img(self, blobs_S, blobs_T, train_op, D_inst_op, D_img_op):
    source_label = 0
    target_label = 1

    train_op.zero_grad()
    # D_inst_op.zero_grad()
    D_img_op.zero_grad()
    
    # sig = nn.Sigmoid()
    bceLoss_func = nn.BCEWithLogitsLoss()

    #train with source
    fc7, net_conv = self.forward(blobs_S['data'], blobs_S['im_info'], blobs_S['gt_boxes'])

    loss_D_img_S = 0
    for idx, n in enumerate(net_conv):
      net_conv[idx] = grad_reverse(n)
      #D_img
      D_img_out = self.D_img(net_conv[idx])
      #loss 
      loss_D_img_S += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(source_label).cuda())
    loss_D_img_S /= len(net_conv)

    #det loss
    loss_S = self._losses['total_loss']
    
    total_loss_S = loss_S + (cfg.ADAPT_LAMBDA/2.) * loss_D_img_S#(loss_D_inst_S + loss_D_img_S + loss_D_const_S)

    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    #train with target
    fc7, net_conv = self.forward(blobs_T['data'], blobs_T['im_info'], blobs_T['gt_boxes'], adapt=True)

    loss_D_img_T = 0
    for idx, n in enumerate(net_conv):
      net_conv[idx] = grad_reverse(n)
      #D_img
      D_img_out = self.D_img(net_conv[idx])
      #loss
      loss_D_img_T += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(target_label).cuda())
    loss_D_img_T /= len(net_conv)

    total_loss_T = (cfg.ADAPT_LAMBDA/2.) * loss_D_img_T#(loss_D_inst_T + loss_D_img_T + loss_D_const_T)

    total_loss = total_loss_S + total_loss_T
    total_loss.backward()

    train_op.step()
    D_img_op.step()
                                                                        
    self.delete_intermediate_states()

    loss_D_inst_S, loss_D_const_S, loss_D_inst_T, loss_D_const_T = 0, 0, 0, 0

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, loss_D_inst_S, loss_D_img_S, loss_D_const_S, loss_D_inst_T, loss_D_img_T, loss_D_const_T

  def FPN_train_adapt_step_img_x5(self, blobs_S, blobs_T, train_op, D_inst_op, D_img_op, D_img_op1, D_img_op2, D_img_op3, D_img_op4):
    source_label = 0
    target_label = 1

    train_op.zero_grad()
    # D_inst_op.zero_grad()
    D_img_op.zero_grad()
    D_img_op1.zero_grad()
    D_img_op2.zero_grad()
    D_img_op3.zero_grad()
    D_img_op4.zero_grad()
    
    bceLoss_func = nn.BCEWithLogitsLoss()

    #train with source
    fc7, net_conv = self.forward(blobs_S['data'], blobs_S['im_info'], blobs_S['gt_boxes'])

    loss_D_img_S = 0

    net_conv[0] = grad_reverse(net_conv[0])
    #D_img
    D_img_out = self.D_img(net_conv[0])
    #loss 
    loss_D_img_S += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(source_label).cuda())

    net_conv[1] = grad_reverse(net_conv[1])
    #D_img
    D_img_out = self.D_img1(net_conv[1])
    #loss 
    loss_D_img_S += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(source_label).cuda())

    net_conv[2] = grad_reverse(net_conv[2])
    #D_img
    D_img_out = self.D_img2(net_conv[2])
    #loss 
    loss_D_img_S += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(source_label).cuda())

    net_conv[3] = grad_reverse(net_conv[3])
    #D_img
    D_img_out = self.D_img3(net_conv[3])
    #loss 
    loss_D_img_S += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(source_label).cuda())

    net_conv[4] = grad_reverse(net_conv[4])
    #D_img
    D_img_out = self.D_img4(net_conv[4])
    #loss 
    loss_D_img_S += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(source_label).cuda())

    loss_D_img_S /= len(net_conv)

    #det loss
    loss_S = self._losses['total_loss']
    
    total_loss_S = loss_S + (cfg.ADAPT_LAMBDA/2.) * loss_D_img_S#(loss_D_inst_S + loss_D_img_S + loss_D_const_S)

    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    #train with target
    fc7, net_conv = self.forward(blobs_T['data'], blobs_T['im_info'], blobs_T['gt_boxes'], adapt=True)

    loss_D_img_T = 0
    net_conv[0] = grad_reverse(net_conv[0])
    #D_img
    D_img_out = self.D_img(net_conv[0])
    #loss
    loss_D_img_T += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(target_label).cuda())

    net_conv[1] = grad_reverse(net_conv[1])
    #D_img
    D_img_out = self.D_img1(net_conv[1])
    #loss
    loss_D_img_T += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(target_label).cuda())

    net_conv[2] = grad_reverse(net_conv[2])
    #D_img
    D_img_out = self.D_img2(net_conv[2])
    #loss
    loss_D_img_T += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(target_label).cuda())

    net_conv[3] = grad_reverse(net_conv[3])
    #D_img
    D_img_out = self.D_img3(net_conv[3])
    #loss
    loss_D_img_T += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(target_label).cuda())

    net_conv[4] = grad_reverse(net_conv[4])
    #D_img
    D_img_out = self.D_img4(net_conv[4])
    #loss
    loss_D_img_T += bceLoss_func(D_img_out, torch.FloatTensor(D_img_out.data.size()).fill_(target_label).cuda())
    

    loss_D_img_T /= len(net_conv)

    total_loss_T = (cfg.ADAPT_LAMBDA/2.) * loss_D_img_T#(loss_D_inst_T + loss_D_img_T + loss_D_const_T)

    total_loss = total_loss_S + total_loss_T
    total_loss.backward()

    train_op.step()
    D_img_op.step()
    D_img_op1.step()
    D_img_op2.step()
    D_img_op3.step()
    D_img_op4.step()
                                                                        
    self.delete_intermediate_states()

    loss_D_inst_S, loss_D_const_S, loss_D_inst_T, loss_D_const_T = 0, 0, 0, 0

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, loss_D_inst_S, loss_D_img_S, loss_D_const_S, loss_D_inst_T, loss_D_img_T, loss_D_const_T

  def train_step(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])

    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]

    #utils.timer.timer.tic('backward')
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    #utils.timer.timer.toc('backward')
    train_op.step()

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_with_summary(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].data[0], \
                                                                        self._losses['rpn_loss_box'].data[0], \
                                                                        self._losses['cross_entropy'].data[0], \
                                                                        self._losses['loss_box'].data[0], \
                                                                        self._losses['total_loss'].data[0]
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    summary = self._run_summary_op()

    self.delete_intermediate_states()

    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

  def train_step_no_return(self, blobs, train_op):
    self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'])
    train_op.zero_grad()
    self._losses['total_loss'].backward()
    train_op.step()
    self.delete_intermediate_states()

  def load_state_dict(self, state_dict):
    """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
    netDict = self.state_dict()
    stateDict = {k: v for k, v in state_dict.items() if k in netDict}
    netDict.update(stateDict)
    nn.Module.load_state_dict(self, netDict)

    # nn.Module.load_state_dict(self, {k: state_dict[k] for k in list(self.state_dict())})
