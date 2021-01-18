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

"""Module for XLayer neural network layers implemented on top of tensorflow"""

import os
import abc
import math
import numpy as np
import tensorflow as tf
import logging

from typing import List

from ..rt_layer_tf import RtLayerTF
from ..x_2_tf_registry import rt_register_xlayer_2_tf,\
    rt_register_xlayer_2_tf_factory_func

from ... import base
from ... import rt_layer

logger = logging.getLogger("pyxir")


#######
# Add #
#######

@rt_register_xlayer_2_tf('Maximum')
class MaximumLayer(rt_layer.BaseLayer, RtLayerTF):
    """Maximum layer with numpy-style broadcasting"""

    def init(self) -> None:
        assert len(self.inputs) == 2
        logger.debug("Maximum START")

        self.left = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.right = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])

        self.inpts = [self.left, self.right]
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.debug("Maximum res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert len(inpts) == 2
        left, right = inpts[0], inpts[1]
        return [tf.maximum(left, right, name=self.name)]

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert len(inputs) == 2
        with tf.compat.v1.Session() as sess:
            feed_dict = {self.inpts[0]: inputs[0], self.inpts[1]: inputs[1]}
            return sess.run(self.res, feed_dict=feed_dict)


########
# Mean #
########

@rt_register_xlayer_2_tf('Mean')
class MeanLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self) -> None:
        self.axes, self.keepdims = \
            self.attrs['axes'], self.attrs['keepdims']

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert len(inpts) == 1, "Mean layer expects one input"
        axes, keepdims = self.axes, self.keepdims
        return [tf.reduce_mean(
            inpts[0],
            axis=list(axes),
            keepdims=keepdims,
            name=self.name
        )]

    def forward_exec(self, inputs: List[np.ndarray]) -> np.ndarray:
        assert len(inputs) == 1, "Mean layer expects one input"
        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})
