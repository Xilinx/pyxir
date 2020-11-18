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

from ..x_2_tf_registry import rt_register_xlayer_2_tf,\
    rt_register_xlayer_2_tf_factory_func

from ..rt_layer_tf import RtLayerTF
from ... import rt_layer
from ... import base

from pyxir.shapes import TupleShape, TensorShape

logger = logging.getLogger("pyxir")

############
# CONSTANT #
############


class ConstantLayer(rt_layer.ConstantLayer, RtLayerTF):

    def init(self):
        # type: (List[int]) -> None
        self.inpt = self.value
        self.res = self.inpt
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 0)

        # dtype = RtLayerTF.dtype_to_tf[self.dtype]
        # return [tf.cast(self.value, dtype)]

        dtype = RtLayerTF.dtype_to_np[self.dtype]
        return [self.value.astype(dtype)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 0)

        return self.res


@rt_register_xlayer_2_tf_factory_func('Constant')
def constant_factory():
    return base.get_constant_layer(ConstantLayer)


#########
# INPUT #
#########


class InputLayer(rt_layer.InputLayer, RtLayerTF):

    def init(self):
        # type: (List[int]) -> None
        self.inpt = self.get_output_tensors([])[0]
        self.res = self.inpt
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 0)

        dtype = RtLayerTF.dtype_to_tf[self.dtype]

        return [tf.compat.v1.placeholder(dtype, shape=self.input_shapes[0],
                                         name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 1)
        # return inputs[0]
        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('Input')
def input_factory():
    return base.get_input_layer(InputLayer, ConstantLayer)


##########
# OUTPUT #
##########

@rt_register_xlayer_2_tf('Output')
class OutputLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        return [tf.identity(inpts[0], name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 1)
        # return inputs[0]
        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

############
# StrInput #
############


@rt_register_xlayer_2_tf('StrInput')
class StrInputLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: (List[int]) -> None
        self.inpt = self.get_output_tensors([])[0]
        self.res = self.inpt
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 0)

        return [tf.compat.v1.placeholder(tf.string, name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 1)
        # return inputs[0]
        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

    def is_input_layer(self):
        return True

#########
# Tuple #
#########


@rt_register_xlayer_2_tf('Tuple')
class TupleLayer(rt_layer.BaseLayer, RtLayerTF):

    """
    Tuple layer takes input layers and groups them in a tuple output
    """

    def init(self):
        logger.debug("Initializing TupleLayer with shape: {}"
                     .format(self.shape))

        self.inpt = []
        for ishape in self.input_shapes:
            plch = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=ishape.tolist())
            self.inpt.append(plch)

        self.res = self.get_output_tensors(self.inpt)[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        inpts = [tf.identity(i) for i in inpts]
        res = tf.tuple(inpts)
        return [res]

    def forward_exec(self, inputs):
        # type: (List[List[str]]) -> numpy.ndarray
        assert(len(inputs) == len(self.inputs))

        with tf.compat.v1.Session() as sess:
            feed_dict = \
                {inpt: inputs[idx] for idx, inpt in enumerate(self.inpt)}
            return sess.run(self.res, feed_dict=feed_dict)

################
# TupleGetItem #
################


@rt_register_xlayer_2_tf('TupleGetItem')
class TupleGetItemLayer(rt_layer.BaseLayer, RtLayerTF):

    """ Tuple layer takes an element from a tuple input """

    def init(self):
        logger.debug("Initializing TupleGetItemLayer with shape: {}"
                     .format(self.shape))

        self.index = self.attrs['index']
        self.transpose = 'transpose' in self.attrs and self.attrs['transpose']
        self.axes = list(self.attrs['axes']) if self.transpose else []

        self.inpt = []
        assert len(self.input_shapes) == 1
        for ishape in self.input_shapes[0]:
            plch = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=ishape.tolist())
            self.inpt.append(plch)

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        res = inpts[0][self.index]

        if self.transpose:
            return [tf.transpose(res, list(self.axes))]

        return [res]

    def forward_exec(self, inputs):
        # type: (List[List[str]]) -> numpy.ndarray
        assert len(inputs) == len(self.inpt)

        with tf.compat.v1.Session() as sess:
            feed_dict = \
                {inpt: inputs[idx] for idx, inpt in enumerate(self.inpt)}
            return sess.run(self.res, feed_dict=feed_dict)

############
# Variable #
############


@rt_register_xlayer_2_tf('Variable')
class VariableLayer(rt_layer.BaseLayer, RtLayerTF):

    constraints = {
        'non-negativity': lambda x: x
    }

    def init(self):
        # type: () -> None
        """
        """
        self.value = self.data[0]
        # self.value = self.attrs['init_value']
        self.trainable = self.attrs['trainable'] if 'trainable' in self.attrs \
            else True
        self.constraint = self.attrs['constraint'] if 'constraint' in \
            self.attrs else None

        if self.constraint is not None and \
                self.constraint not in VariableLayer.constraints:
            raise ValueError("Unrecognized constraint: {}, the possible"
                             " constraints are {}"
                             .format(constraint,
                                     VariableLayer.constraints.keys()))

        with tf.name_scope('init'):
            self.res = self.get_output_tensors([], force_trainable=False)[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, force_trainable=None, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 0)

        trainable = self.trainable if force_trainable is None else\
            force_trainable
        constraint = VariableLayer.constraints[self.constraint] \
            if self.constraint is not None else None
        logger.debug("Variable constraint: {}".format(constraint))

        # TODO
        # if constraint == 'non-negativity':
        #    initial_value = np.log(self.value)
        # else:
        #    initial_value = self.value
        initial_value = self.value

        var = tf.Variable(
            initial_value=np.atleast_1d(initial_value).astype(np.float32),
            trainable=trainable,
            name=self.name,
            constraint=constraint
        )

        # if constraint == 'non-negativity':
        #    var = tf.exp(var)

        return [var]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 0)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            return sess.run(self.res, feed_dict={})
