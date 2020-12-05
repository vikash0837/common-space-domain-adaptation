# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math

from utils.timer import Timer
from model.nms_wrapper import nms
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv

import torch
import json
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
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

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois, fc7, net_conv = net.test_image(blobs['data'], blobs['im_info'])

  boxes = rois[:, 1:5] / im_scales[0]
  scores = np.reshape(scores, [scores.shape[0], -1])
  bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
    pred_boxes = _clip_boxes(pred_boxes, im.shape)
  else:
    # Simply repeat the boxes, once for each class
    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

  return scores, pred_boxes#, fc7, net_conv
  #return scores, pred_boxes,fc7, net_conv

def apply_nms(all_boxes, thresh):
  """Apply non-maximum suppression to all predicted boxes output by the
  test_net method.
  """
  num_classes = len(all_boxes)
  num_images = len(all_boxes[0])
  nms_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
  for cls_ind in range(num_classes):
    for im_ind in range(num_images):
      dets = all_boxes[cls_ind][im_ind]
      if dets == []:
        continue

      x1 = dets[:, 0]
      y1 = dets[:, 1]
      x2 = dets[:, 2]
      y2 = dets[:, 3]
      scores = dets[:, 4]
      inds = np.where((x2 > x1) & (y2 > y1))[0]
      dets = dets[inds,:]
      if dets == []:
        continue

      keep = nms(torch.from_numpy(dets), thresh).numpy()
      if len(keep) == 0:
        continue
      nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
  return nms_boxes
  
def draw_car_bb(im, bboxes, scores=[], thr=0.3, color_type='1'):
    bboxes = bboxes.astype(int)
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape

    
    if color_type == 'gt':
      scores = np.ones(len(bboxes))
      color = (0,0,255)
    elif color_type == '1':
      color = (255,0,0)
    elif color_type == '2':
      color = (0,255,0)

    for i, box in enumerate(bboxes):
      if scores[i] < thr:
          continue

      thick = int((h + w) / 1000) #original: int((h + w) / 300)
      cv2.rectangle(imgcv,
                    (box[0], box[1]), (box[2], box[3]),
                    color, thick)
      mess = '%s: %.3f' % ('Car', scores[i])
      if color_type == 'gt':
        mess = ''
      cv2.putText(imgcv, mess, (box[0], box[1] - 12),
                  0, 1e-3 * h / 2., color, 2)

    return imgcv

def split_bbox(bbox, imgname, class_recs):
  R = class_recs[imgname]
  BBGT = R['bbox'].astype(float)

  ov_th = []
  und_th = []
  gt_left = np.ones(len(BBGT))
  for bb in bbox:
    assert bb.shape[0] == 5
    ovmax = -np.inf
    bb = bb.astype(float)
    if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        gt_left[jmax] = 0

    if ovmax >= 0.5:
      ov_th.append(bb)
    else:
      und_th.append(bb)

  gt_left = np.where(gt_left == 1)[0]

  return np.array(ov_th), np.array(und_th), BBGT[gt_left] # N, box+score
def get_sample_score(DT_score, score):
  "DT_score is scalar and score are numpy array"
  div_que = (1-DT_score)/DT_score
  uncertain_que = -1.0*score*np.log2(score)
  uncertain_que = np.sum(uncertain_que)
  return div_que,div_que*uncertain_que

def test_discriminator(net, imdb, weights_filename, max_per_image=100, thresh=0.):
  vis = False

  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  print("num_images=:",num_images)
  #_, scores, bbox_pred, rois, fc7, net_conv = net.test_image(blobs['data'], blobs['im_info'])
  diversity_score = {}
  div_and_uncertain_score = {}
  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    print("processing for:",imdb.image_path_at(i))
    blobs, im_scales = _get_blobs(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
    _, cls_prob, bbox_pred, D_T_score,net_conv = net.test_discriminator(im_blob,blobs['im_info'])
    # print("######3 printing result #########")
    # print("D_T_score:",D_T_score)
    
    # print("D_T_score shape=:",D_T_score.data.size())
    # print("net_conv shape:",net_conv.shape)

    #print("cls_prob shape=:{} and cls_prob={}".format(cls_prob.shape,cls_prob))
    dt_value  = D_T_score.cpu().detach().numpy()[0]
    score = cls_prob
    div_que_score,product_score  = get_sample_score(dt_value,score)
    print("Diversity score=:{} and total_score=:{}".format(div_que_score,product_score))
    image_name = imdb.image_path_at(i).split("/")[-1]
    diversity_score[image_name] = div_que_score
    div_and_uncertain_score[image_name] = product_score
  with open("diversity_score.json",'w') as fp:
    json.dump(diversity_score,fp)
    fp.close()
  with open("div_and_uncertain_score.json",'w') as fp:
    json.dump(div_and_uncertain_score,fp)
    fp.close()
  print("files written successfully")

    #synth_weight = imdb.D_T_score["munster_000038_000019_leftImg8bit.png"]
    #print("synth_weight:",synth_weight)
    #break





def test_net(net, imdb, weights_filename, max_per_image=100, thresh=0.):
  vis = False

  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  ##
  original_all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  ##

  output_dir = get_output_dir(imdb, weights_filename)

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  # extract gt objects for this class
  class_recs = {}
  npos = 0

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    _t['im_detect'].tic()
    scores, boxes = im_detect(net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()

    # skip j = 0, because it's the background class
    for j in range(1, imdb.num_classes):
      inds = np.where(scores[:, j] > thresh)[0]
      cls_scores = scores[inds, j]
      cls_boxes = boxes[inds, j*4:(j+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
      keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).cpu().numpy() if cls_dets.size > 0 else []
      # ##
      # original_all_boxes[j][i] = cls_dets
      # ##
      cls_dets = cls_dets[keep, :]
      all_boxes[j][i] = cls_dets
    ##
    obj_scores = net.roi_scores.cpu().data.numpy()
    inds = np.where(obj_scores[:] > thresh)[0]
    cls_scores = obj_scores[inds]
    cls_boxes = boxes[inds, 4:8]
    cls_dets = np.hstack((cls_boxes, obj_scores[:])) \
      .astype(np.float32, copy=False)
    
    original_all_boxes[j][i] = cls_dets

    # Limit to max_per_image detections *over all classes*
    if max_per_image > 0:
      image_scores = np.hstack([all_boxes[j][i][:, -1]
                    for j in range(1, imdb.num_classes)])
      if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        for j in range(1, imdb.num_classes):
          keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
          all_boxes[j][i] = all_boxes[j][i][keep, :]
          
    _t['misc'].toc()
    
    print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
        .format(i + 1, num_images, _t['im_detect'].average_time(),
            _t['misc'].average_time()))

  det_file = os.path.join(output_dir, 'detections.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

#@modified_method
def get_bbox_for_max_score(im,scores,boxes,img_name,thresh=0.5):
  "scores shape=(n,9), bbox shape=:(n,4*9)"
  image = im
  boxes = np.floor(boxes)
  result = {}
  result["objects"] = []
  result["imgHeight"] = 1024
  result["imgWidth"] = 2048
  #_classes = ('__background__','bicycle','bus','car','motorcycle','person','rider','train','truck')
  _classes = ('__background__', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
  for j in range(1,9):
    if(np.max(scores[:,j]) > thresh):
      idx = np.argmax(scores[:,j])
      #idxs = np.where(scores[:,j] > thresh)
      #print("current IDX",idx)
      #print("boxes shape=:",boxes.shape)
      b_box = boxes[idx,4*j:4*(j+1)]
      #print("###########my b_box",b_box)
      result["objects"].append({"label":_classes[j],"polygon":[[b_box[0],b_box[1]],[b_box[2],b_box[3]]]})
      # start_point = (b_box[0],b_box[1])
      # end_point = (b_box[2],b_box[3])
      # # Blue color in BGR 
      # color = (255, 0, 0) 
      # # Line thickness of 2 px 
      # thickness = 2
      # image = cv2.rectangle(image, start_point, end_point, color, thickness)
  #cv2.imwrite(str(img_name),image)
  #print(result)
  return result






def _test_net(net, imdb, weights_filename, max_per_image=100, thresh=0.):
  vis = False
  print("Numeber of classes=:",imdb.num_classes)
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  # all detections are collected into:
  #  all_boxes[cls][image] = N x 5 array of detections in
  #  (x1, y1, x2, y2, score)
  all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  ##
  original_all_boxes = [[[] for _ in range(num_images)]
         for _ in range(imdb.num_classes)]
  ##

  output_dir = get_output_dir(imdb, weights_filename)

  # timers
  _t = {'im_detect' : Timer(), 'misc' : Timer()}

  # extract gt objects for this class
  class_recs = {}
  npos = 0

  for i in range(num_images):
    im = cv2.imread(imdb.image_path_at(i))
    print("img path=:",imdb.image_path_at(i))
    img_name = imdb.image_path_at(i).split('/')[-1]
    folder_name = imdb.image_path_at(i).split('/')[-2]
    _t['im_detect'].tic()
    scores, boxes = im_detect(net, im)
    _t['im_detect'].toc()

    _t['misc'].tic()
    #print("############################")
    #print("score shape, box shape",scores.shape, boxes.shape)
    result = get_bbox_for_max_score(im,scores,boxes,img_name)
    basepath = "data/temp/foggytrain2"
    filepath = os.path.join(basepath,folder_name)
    if(not os.path.exists(filepath)):
      os.makedirs(filepath)
    json_name = img_name.split(".png")[0] + ".json"
    json_file = os.path.join(filepath,json_name)
    #print("object type",type(result))
    with open(json_file, 'w') as fp:
      json.dump(str(result), fp)
    print("file written at path:",json_file)

  #   break

  #   # skip j = 0, because it's the background class
  #   for j in range(1, imdb.num_classes):
  #     inds = np.where(scores[:, j] > thresh)[0]
  #     cls_scores = scores[inds, j]
  #     cls_boxes = boxes[inds, j*4:(j+1)*4]
  #     cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
  #       .astype(np.float32, copy=False)
  #     keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).cpu().numpy() if cls_dets.size > 0 else []
  #     # ##
  #     # original_all_boxes[j][i] = cls_dets
  #     # ##
  #     cls_dets = cls_dets[keep, :]
  #     all_boxes[j][i] = cls_dets
  #   ##
  #   obj_scores = net.roi_scores.cpu().data.numpy()
  #   inds = np.where(obj_scores[:] > thresh)[0]
  #   cls_scores = obj_scores[inds]
  #   cls_boxes = boxes[inds, 4:8]
  #   cls_dets = np.hstack((cls_boxes, obj_scores[:])) \
  #     .astype(np.float32, copy=False)
    
  #   original_all_boxes[j][i] = cls_dets

  #   # Limit to max_per_image detections *over all classes*
  #   if max_per_image > 0:
  #     image_scores = np.hstack([all_boxes[j][i][:, -1]
  #                   for j in range(1, imdb.num_classes)])
  #     if len(image_scores) > max_per_image:
  #       image_thresh = np.sort(image_scores)[-max_per_image]
  #       for j in range(1, imdb.num_classes):
  #         keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
  #         all_boxes[j][i] = all_boxes[j][i][keep, :]
          
  #   _t['misc'].toc()
    
  #   print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
  #       .format(i + 1, num_images, _t['im_detect'].average_time(),
  #           _t['misc'].average_time()))

  # det_file = os.path.join(output_dir, 'detections.pkl')
  # with open(det_file, 'wb') as f:
  #   pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

  # print('Evaluating detections')
  # imdb.evaluate_detections(all_boxes, output_dir)