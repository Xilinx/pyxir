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
Module for segmentation related pre- and postprocessing functions
"""

import os
import cv2
import numpy as np


def fpn_cityscapes_postp(predictions,
                         result_img_path='result_img',
                         **kwargs):
    label_to_color = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220,  0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255,  0,  0],
        [0,  0, 142],
        [0,  0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        ])
    image_path = kwargs['image_list'][kwargs['idx']]
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    pred = predictions[0]
    pred = pred.argmax(axis=-1)[0]
    gray = cv2.resize(pred, dsize=(width, height),
                      interpolation=cv2.INTER_NEAREST)
    # pred_color = label_img_to_color(pred)
    pred_color = label_to_color[gray].astype('uint8')
    if not os.path.exists(result_img_path):
        os.makedirs(result_img_path)

    cv2.imwrite(
        os.path.join(
            result_img_path,
            'result_{}'.format(os.path.basename(image_path))),
        pred_color)

    return ''
