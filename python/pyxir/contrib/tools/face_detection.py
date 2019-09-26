# Copyright 2020 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Module for face detection related pre- and postprocessing functions
"""

import os
import cv2
import numpy as np


def softmax_cpu(x, dim=-1):
    x = np.exp(x)
    s = np.expand_dims(np.sum(x, axis=dim), dim)
    return x/s


def gstiling(data, stride, reverse=True):
    n, h, w, c = data.shape
    ss = stride * stride
    assert c % ss == 0
    assert reverse  # Only support reverse=True for densebox

    out_c = c // ss
    out_h = h * stride
    out_w = w * stride
    output = data.reshape((n, h, w, stride, stride, out_c))\
        .transpose([0, 1, 3, 2, 4, 5])\
        .reshape((n, out_h, out_w, out_c))
    return output


def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def face_detection_postp(predictions, **kwargs):
    expand_scale_ = 0.0
    res_stride_ = 4
    det_threshold_ = 0.7
    nms_threshold_ = 0.3
    input_channels_ = 3

    ################
    pixel_output, bb_output = predictions
    assert pixel_output.shape[0] == 1  # Only support batch 1

    prob = softmax_cpu(gstiling(pixel_output, 8))
    bb = gstiling(bb_output, 8)

    sz = prob.shape[2]*res_stride_, prob.shape[1] * res_stride_  # w, h

    image_path = kwargs['image_list'][kwargs['idx']]
    image = cv2.imread(image_path)
    image_resize = cv2.resize(image, sz)
    gy = np.arange(0, sz[0], res_stride_)
    gx = np.arange(0, sz[1], res_stride_)
    gy = gy[0: bb.shape[1]]
    gx = gx[0: bb.shape[2]]
    [x, y] = np.meshgrid(gx, gy)

    bb[:, :, :, 0] += x
    bb[:, :, :, 2] += x
    bb[:, :, :, 1] += y
    bb[:, :, :, 3] += y
    bb = np.reshape(bb, (-1, 4))
    prob = np.reshape(prob[:, :, :, 1], (-1, 1))
    bb = bb[prob.ravel() > det_threshold_, :]
    prob = prob[prob.ravel() > det_threshold_, :]
    rects = np.hstack((bb, prob))
    keep = nms(rects, nms_threshold_)
    rects = rects[keep, :]
    rects_expand = []
    for rect in rects:
        rect_expand = []
        rect_w = rect[2]-rect[0]
        rect_h = rect[3]-rect[1]
        rect_expand.append(int(max(0, rect[0]-rect_w*expand_scale_)))
        rect_expand.append(int(max(0, rect[1]-rect_h*expand_scale_)))
        rect_expand.append(int(min(sz[1], rect[2]+rect_w*expand_scale_)))
        rect_expand.append(int(min(sz[0], rect[3]+rect_h*expand_scale_)))
        rects_expand.append(rect_expand)
    for face_rect in rects_expand:
        cv2.rectangle(image_resize, (face_rect[0], face_rect[1]),
                      (face_rect[2], face_rect[3]), (0, 255, 0), 2)
    if not os.path.exists('result_img'):
        os.makedirs('result_img')
    cv2.imwrite('result_img/result_{}'
                .format(os.path.basename(image_path)), image_resize)

    return rects_expand
