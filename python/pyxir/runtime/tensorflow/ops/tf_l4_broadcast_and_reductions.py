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

########
# Mean #
########


@rt_register_xlayer_2_tf('Mean')
class MeanLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: (List[int], List[int], bool, bool) -> None

        self.axes, self.keepdims = \
            self.attrs['axes'], self.attrs['keepdims']

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.res = self.get_output_tensors([self.inpt])[0]

        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        axes, keepdims = self.axes, self.keepdims

        return [tf.reduce_mean(
            inpts[0],
            axis=list(axes),
            keepdims=keepdims
        )]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})
