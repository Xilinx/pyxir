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
# Cast #
########

@rt_register_xlayer_2_tf('Cast')
class CastLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        Initialize a Cast layer on top of tf.cast operation
        """
        self.target_dtype = self.attrs['dtype']

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 1

        return [tf.cast(inpts[0], dtype=self.target_dtype, name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert len(inputs) == 1

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

#############
# LeakyRelu #
#############


@rt_register_xlayer_2_tf('LeakyReLU')
class LeakyReluLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        Initialize a leaky relu layer on top of tf.nn.leaky_relu operation

        y = alpha*x for x < 0
        y = x for x> 0
        """
        self.alpha = self.attrs['alpha']

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        features, alpha = inpts[0], self.alpha

        with tf.name_scope(self.name, "LeakyRelu", [features, alpha]) as name:
            features = tf.convert_to_tensor(features, name="features")
            if features.dtype.is_integer:
                features = tf.to_float(features)
            alpha = tf.convert_to_tensor(alpha, dtype=features.dtype,
                                         name="alpha")
        return [tf.maximum(alpha * features, features, name=name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf('pReLU')
class PReluLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        Initialize a leaky relu layer on top of tf.nn.leaky_relu operation

        y = alpha*x for x < 0
        y = x for x> 0
        """
        self.alpha = self.attrs['alpha']

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        return [tf.nn.leaky_relu(inpts[0], alpha=self.alpha)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

###########
# Reshape #
###########


@rt_register_xlayer_2_tf('Reshape')
class ReshapeLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None

        self.target_shape = self.attrs['shape']

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Input shape: {}".format(self.inpt.shape))
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        input_shape, shape = self.input_shapes[0], self.target_shape
        logger.debug("New shape: {}".format(shape))
        if input_shape[0] in [-1, None] and shape[0] != -1:
            logger.warn("[WARNING]: Manually fixing invalid fixed reshape"
                        " layer shape: {}, input has variable shape in first"
                        " layer".format(shape))
            assert(len(shape) >= 2)
            shape = [-1] + shape[1:]

        return [tf.reshape(
            inpts[0],
            shape=list(shape),
            name=self.name
        )]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

#########
# Split #
#########


@rt_register_xlayer_2_tf('Split')
class SplitLayer(rt_layer.BaseLayer, RtLayerTF):

    """ Split an input tensor along axis and according to provided indeices """

    def init(self):
        logger.debug("Initializing SplitLayer with shape: {}"
                     .format(self.shape))

        self.axis = self.attrs['axis']
        self.indices = self.attrs['indices']

        if isinstance(self.indices, int):
            self.num_or_size_splits = self.indices
        else:
            axis_size = self.input_shapes[0][self.axis]
            prev = 0
            self.num_or_size_splits = []
            for i in self.indices:
                self.num_or_size_splits.append(i - prev)
                prev = i
            self.num_or_size_splits.append(axis_size - prev)

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        res = tf.split(inpts[0], self.num_or_size_splits, axis=self.axis)
        return [res]

    def forward_exec(self, inputs):
        # type: (List[List[str]]) -> numpy.ndarray
        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

###########
# Squeeze #
###########


@rt_register_xlayer_2_tf('Squeeze')
class SqueezeLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None

        self.axis = list(self.attrs['axis'])

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Input shape: {}".format(self.inpt.shape))
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        return [tf.squeeze(
            inpts[0],
            axis=self.axis
        )]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

########
# Take #
########


@rt_register_xlayer_2_tf('Take')
class TakeLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None

        self.axis = self.attrs['axis']
        self.mode = self.attrs['mode']

        # TODO
        assert self.mode == 'clip'

        inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        # TODO: input dtypes
        indices = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf['int32'],
                                     shape=self.input_shapes[1])

        self.inpts = [inpt, indices]

        self.res = self.get_output_tensors(self.inpts)[0]
        logger.info("Input shape: {}".format(inpt.shape))
        logger.info("Indices shape: {}".format(indices.shape))
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 2

        return [tf.gather(inpts[0], inpts[1], axis=self.axis)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == len(self.input_shapes))
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)

#############
# Transpose #
#############


@rt_register_xlayer_2_tf('Transpose')
class TransposeLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.axes = self.attrs['axes']

        logger.debug("Transpose layer axes: {}".format(self.axes))
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Input shape: {}".format(self.inpt.shape))
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        return [tf.transpose(inpts[0], list(self.axes))]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})
