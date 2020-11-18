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
import typing
import logging

import numpy as np
import tensorflow as tf

from .tf_l0_input_and_other import ConstantLayer

from ..rt_layer_tf import RtLayerTF
from ..x_2_tf_registry import rt_register_xlayer_2_tf,\
    rt_register_xlayer_2_tf_factory_func

from ... import base
from ... import rt_layer

logger = logging.getLogger("pyxir")


# TODO
########
# Relu #
########

@rt_register_xlayer_2_tf('ReLU')
class ReluLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        Initialize a relu layer on top of tf.relu operation
        """
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, override_name=None, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        name = self.name if override_name is None else override_name
        return [tf.nn.relu(inpts[0], name=name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


#######
# Add #
#######

@rt_register_xlayer_2_tf('Add')
class AddLayer(rt_layer.BaseLayer, RtLayerTF):

    """ Add layer with numpy-style broadcasting """

    def init(self):
        # type: () -> None
        assert len(self.inputs) == 2
        logger.debug("Add START")

        self.left = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.right = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])

        self.inpts = [self.left, self.right]
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.debug("Add res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 2
        left, right = inpts[0], inpts[1]

        return [tf.add(left, right)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert len(inputs) == 2

        with tf.compat.v1.Session() as sess:
            feed_dict = {self.inpts[0]: inputs[0], self.inpts[1]: inputs[1]}
            return sess.run(self.res, feed_dict=feed_dict)


###########
# BiasAdd #
###########

class BiasAddLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        assert len(self.inputs) == 2
        logger.debug("BiasAdd START")

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.bias = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])

        self.res = self.get_output_tensors([self.inpt, self.bias])[0]
        logger.debug("BiasAdd res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 2
        inpt, bias = inpts[0], inpts[1]

        if inpt.dtype not in ['float32', tf.float32]:
            inpt = tf.cast(inpt, RtLayerTF.dtype_to_tf[self.dtype])
        if bias.dtype not in ['float32', tf.float32]:
            bias = tf.cast(bias, RtLayerTF.dtype_to_tf[self.dtype])

        axis = self.attrs['axis']
        logger.debug("Axis: {}".format(axis))

        if inpt.shape != bias.shape and axis not in [None, -1]:
            shape_for_broadcast = [(1 if i != axis else bias.shape[0])
                                   for i in range(len(inpt.shape))]
            bias = tf.reshape(bias, tuple(shape_for_broadcast))
        # Else tf broadcasting

        return [tf.add(inpt, bias)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert len(inputs) == 2

        with tf.compat.v1.Session() as sess:
            feed_dict = {self.inpt: inputs[0], self.bias: inputs[1]}
            return sess.run(self.res, feed_dict=feed_dict)


@rt_register_xlayer_2_tf_factory_func('BiasAdd')
def bias_add_factory():
    return base.get_bias_add_layer(BiasAddLayer, ConstantLayer)

##########
# Concat #
##########


@rt_register_xlayer_2_tf('Concat')
class ConcatLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.axis = self.attrs['axis']

        self.inpts = []
        for shape in self.input_shapes:
            plch = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=shape)
            self.inpts.append(plch)

        self.res = self.get_output_tensors(self.inpts)[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        return [tf.concat(inpts, axis=self.axis, name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == len(self.inputs))

        with tf.compat.v1.Session() as sess:
            feed_dict = {inpt: inputs[idx] for idx, inpt in
                         enumerate(self.inpts)}
            return sess.run(self.res, feed_dict=feed_dict)

#########
# Dense #
#########


class DenseLayer(rt_layer.DenseLayer, RtLayerTF):

    def init(self):
        # type: () -> None

        inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        input_shapes, weights, biases = \
            self.input_shapes, self.weights, self.biases
        if weights is not None:
            weights = \
                tf.compat.v1.placeholder_with_default(weights, weights.shape)
            weights = tf.cast(weights, RtLayerTF.dtype_to_tf[self.dtype])
        else:
            weights = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[1])

        if biases is not None:
            biases = \
                tf.compat.v1.placeholder_with_default(biases, biases.shape)
            biases = tf.cast(biases, RtLayerTF.dtype_to_tf[self.dtype])
        else:
            biases = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[2])
        self.inpts = [inpt, weights, biases]
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.debug("Dense layer: {}, res shape: {}"
                     .format(self.name, self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 3)

        inpt, weights, biases = inpts

        input_shape = self.input_shapes[0]

        if self.kernel_layout == 'OI':
            weights = tf.transpose(weights, (1, 0))

        inpt = tf.expand_dims(inpt, 0) if len(input_shape) == 1 else inpt

        res = tf.add(tf.matmul(inpt, weights), biases)

        if self.use_relu:
            res = tf.nn.relu(res)

        if len(self.shape) == 1:
            res = tf.squeeze(res)

        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == len(self.input_shapes))
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)

    def get_params(self):
        return {'W': self.weights, 'B': self.biases}


@rt_register_xlayer_2_tf_factory_func('Dense')
def dense_factory():
    return base.get_dense_layer(DenseLayer, ConstantLayer)

###########
# Eltwise #
###########


class ElementwiseLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.op = self.attrs['op']

        if len(self.inputs) != 2:
            raise ValueError("Run elementwise operation expects 2 inputs, {}"
                             " given".format(len(self.inputs)))

        if self.op != 'Add':
            raise NotImplementedError("Only elementwise add operation"
                                      " supported at the"
                                      " moment, not: {}".format(self.op))

        left = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        right = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])

        self.inpts = [left, right]
        self.res = self.get_output_tensors(self.inpts)[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 2

        # Do calculation in float32
        inpt0, inpt1 = tf.cast(inpts[0], tf.float32), \
            tf.cast(inpts[1], tf.float32)

        return [tf.add(inpt0, inpt1, name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == len(self.inputs))

        with tf.compat.v1.Session() as sess:
            feed_dict = {inpt: inputs[idx] for idx, inpt in
                         enumerate(self.inpts)}
            return sess.run(self.res, feed_dict=feed_dict)


@rt_register_xlayer_2_tf_factory_func('Eltwise')
def eltwise_factory():
    return base.get_elementwise_layer(ElementwiseLayer, ReluLayer)


#######
# Exp #
#######

@rt_register_xlayer_2_tf('Exp')
class ExpLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """ Initialize a exponent layer on top of tf.exp operation """
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        return [tf.exp(inpts[0], name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


##############
# ExpandDims #
##############

@rt_register_xlayer_2_tf('ExpandDims')
class ExpandDimsLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 1

        new_shape = [d if d is not None else -1 for d in self.shape[:]]

        return [tf.reshape(inpts[0], shape=new_shape, name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

#######
# Pad #
#######


@rt_register_xlayer_2_tf('Pad')
class PadLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        logger.debug("Pad layer: {}".format(self.attrs['padding']))
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        paddings = [list(pad) for pad in self.attrs['padding']]

        return [tf.pad(inpts[0], paddings=paddings, mode="CONSTANT", name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


#########
# Relu6 #
#########

@rt_register_xlayer_2_tf('ReLU6')
class Relu6Layer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        Initialize a relu6 layer on top of tf.nn.relu6 operation
        """
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 1
        return [tf.nn.relu6(inpts[0], name=self.name)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

#########
# Scale #
#########


class ScaleLayer(rt_layer.ScaleLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.axis = self.attrs['axis']

        logger.info("Scale layer, axis: {}".format(self.axis))

        input_shapes, gamma, beta = self.input_shapes, self.gamma, self.beta
        inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=input_shapes[0])

        if gamma is not None:
            gamma = tf.compat.v1.placeholder_with_default(gamma, gamma.shape)
        else:
            gamma = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[1])

        if beta is not None:
            beta = tf.compat.v1.placeholder_with_default(beta, beta.shape)
        else:
            beta = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[2])

        self.inpts = [inpt, gamma, beta]
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 3)

        inpt, gamma, beta = inpts
        assert gamma.shape == beta.shape

        if self.dtype not in ['float32']:
            inpt = tf.cast(inpt, RtLayerTF.dtype_to_tf[self.dtype])
            gamma = tf.cast(gamma, RtLayerTF.dtype_to_tf[self.dtype])
            beta = tf.cast(beta, RtLayerTF.dtype_to_tf[self.dtype])

        if self.axis not in [None, -1]:
            shape = [(1 if i != self.axis else -1)
                     for i in range(len(self.shape))]
            gamma, beta = tf.reshape(gamma, shape), tf.reshape(beta, shape)
        
        if len(gamma.shape) == 0:
            gamma = np.reshape(gamma, (1,)) if isinstance(gamma, np.ndarray) else tf.reshape(gamma, (1,))
            beta = np.reshape(beta, (1,)) if isinstance(beta, np.ndarray) else tf.reshape(beta, (1,))

        compiler_target = kwargs['compiler_target'] if 'compiler_target' in kwargs else None
        if compiler_target == 'DPUv1Compiler':
            return [tf.add(
                tf.multiply(inpt, gamma, name=self.name),
                beta,
                name=self.name + "/Add"
            )]
        else:
            return [tf.add(
                tf.multiply(inpt, gamma),
                beta,
                name=self.name
            )]
        # return [tf.nn.batch_normalization(
        #     inpt,
        #     mean=tf.zeros(beta.shape),
        #     variance=tf.ones(beta.shape),
        #     offset=beta,
        #     scale=gamma,
        #     variance_epsilon=0.000001,
        #     name=self.name
        # )]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == len(self.input_shapes))
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)


@rt_register_xlayer_2_tf_factory_func('Scale')
def dense_factory():
    return base.get_scaling_layer(ScaleLayer, ConstantLayer, ReluLayer)

###########
# Sigmoid #
###########


@rt_register_xlayer_2_tf('Sigmoid')
class SigmoidLayer(rt_layer.BaseLayer, RtLayerTF):

    """ Sigmoid: y = 1 / (1 + exp(-x)) """

    def init(self):
        # type: () -> None
        """ Initialize a sigmoid layer on top of tf.nn.sigmoid operation
        """
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> List[tf.Tensor]
        """ Return Tensorflow sigmoid computation op """

        assert len(inpts) == 1

        return [tf.nn.sigmoid(inpts[0])]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert len(inputs) == 1

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


###########
# Softmax #
###########

@rt_register_xlayer_2_tf('Softmax')
class SoftmaxLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        return [tf.nn.softmax(inpts[0])]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


#######
# Sub #
#######

@rt_register_xlayer_2_tf('Sub')
class SubLayer(rt_layer.BaseLayer, RtLayerTF):

    """ Subtract layer with numpy-style broadcasting """

    def init(self):
        # type: () -> None
        assert len(self.inputs) == 2
        logger.debug("Tf Sub init")

        self.left = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.right = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])

        self.inpts = [self.left, self.right]
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.debug("Sub res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 2
        left, right = inpts[0], inpts[1]

        return [tf.subtract(left, right)]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert len(inputs) == 2

        with tf.compat.v1.Session() as sess:
            feed_dict = {self.inpts[0]: inputs[0], self.inpts[1]: inputs[1]}
            return sess.run(self.res, feed_dict=feed_dict)


########
# Tanh #
########

@rt_register_xlayer_2_tf('Tanh')
class TanhLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        Initialize a tanh layer on top of tf.tanh operation
        """
        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        return [tf.tanh(inpts[0])]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})
