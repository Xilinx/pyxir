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
Module for XLayer neural network layers implemented on top of tensorflow


"""

import os
import abc
import math
import numpy as np
import tensorflow as tf
import logging

from ..rt_layer_tf import RtLayerTF
from ..x_2_tf_registry import rt_register_xlayer_2_tf,\
    rt_register_xlayer_2_tf_factory_func

from ... import base
from ... import rt_layer

logger = logging.getLogger("pyxir")


@rt_register_xlayer_2_tf('Cvx')
class CvxLayer(rt_layer.BaseLayer, RtLayerTF):

    """
    Cvx layer which takes in a list of strings representing
    the image paths and subsequently loads the images and performs
    specified preprocessing.
    """

    def init(self):
        from pyxir.io.cvx import ImgLoader, ImgProcessor

        self.ImgLoader = ImgLoader
        self.ImgProcessor = ImgProcessor

        logger.debug("Initializing CvxLayer with shape: {}"
                     .format(self.shape))

        self.cvx_key = self.attrs['cvx_key']

        self.inpt = tf.compat.v1.placeholder(tf.string)
        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        def cvx_func(str_lst):
            str_lst = [e.decode('utf-8') for e in str_lst]

            img_loader = self.ImgLoader()
            img_processor = self.ImgProcessor(
                proc_key=self.cvx_key
            )

            data = img_loader.load(str_lst)

            return img_processor.execute(data)

        # Should be in NHWC
        res = tf.py_func(cvx_func, [inpts[0]], tf.float32)

        res_shape = self.shape
        res.set_shape(res_shape)

        return [res]

    def forward_exec(self, inputs):
        # type: (List[List[str]]) -> numpy.ndarray
        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})
