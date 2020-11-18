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

from .tf_l0_input_and_other import ConstantLayer
from .tf_l1_basic_nn import ReluLayer

from ..rt_layer_tf import RtLayerTF
from ..x_2_tf_registry import rt_register_xlayer_2_tf,\
    rt_register_xlayer_2_tf_factory_func

from ... import base
from ... import rt_layer

logger = logging.getLogger("pyxir")

#############
# BatchNorm #
#############


class BatchNormLayer(rt_layer.BatchNormLayer, RtLayerTF):

    def init(self):
        # type: () -> None

        self.axis = self.attrs['axis']

        input_shapes, mu, variance, gamma, beta = \
            self.input_shapes, self.mean, self.variance, self.gamma, self.beta
        inpt = tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                        shape=input_shapes[0])

        if mu is not None:
            mu = tf.compat.v1.placeholder_with_default(mu, mu.shape)
        else:
            mu = tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                          shape=input_shapes[1])

        if variance is not None:
            variance = \
                tf.compat.v1.placeholder_with_default(variance, variance.shape)
        else:
            variance = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[2])

        if gamma is not None:
            gamma = \
                tf.compat.v1.placeholder_with_default(gamma, gamma.shape)
        else:
            gamma = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[3])

        if beta is not None:
            beta = tf.compat.v1.placeholder_with_default(beta, beta.shape)
        else:
            beta = tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                            shape=input_shapes[4])

        self.inpts = [inpt, mu, variance, gamma, beta]
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 5

        inpt, mean, variance, gamma, beta = inpts

        shape = [(1 if i != self.axis else -1) for i in range(len(self.shape))]
        logger.debug("BatchNorm mean/variance shape: {}".format(shape))

        return [tf.nn.batch_normalization(
            inpt,
            mean=tf.reshape(mean, shape),
            variance=tf.reshape(variance, shape),
            offset=tf.reshape(beta, shape),
            scale=tf.reshape(gamma, shape),
            variance_epsilon=self.variance_epsilon
        )]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == len(self.input_shapes))

        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)


@rt_register_xlayer_2_tf_factory_func('BatchNorm')
def bias_add_factory():
    return base.get_batchnorm_layer(BatchNormLayer, ConstantLayer, ReluLayer)

###############
# Convolution #
###############


class ConvLayer(rt_layer.ConvLayer, RtLayerTF):

    def init(self):
        # () -> None
        """
        Initialize a convolution layer on top of tf.nn.conv2d operation
        """
        self.layout = self.attrs['data_layout']

        logger.info("Init {} layer: {}".format(
            'Conv2D' if self.kernel_groups == 1 else 'DepthWiseConv2D',
            self.layout))
        logger.info(self.input_shapes)

        input_shapes, kernel, biases = \
            self.input_shapes, self.kernel, self.biases
        if kernel is not None:
            kernel = \
                tf.compat.v1.placeholder_with_default(kernel, kernel.shape)
        else:
            kernel = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[1])

        if biases is not None:
            biases = \
                tf.compat.v1.placeholder_with_default(biases, biases.shape)
        else:
            biases = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[2])

        inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=input_shapes[0])

        self.inpts = [inpt, kernel, biases]

        self.quant_output = self._get_conv_tensor(self.inpts)
        self.res = self.get_output_tensors(self.inpts)[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def _get_conv_tensor(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor

        if len(inpts) == 3:
            inpt, kernel, biases = inpts
        elif len(inpts) == 1:
            inpt = inpts[0]
            kernel = \
                tf.compat.v1.placeholder_with_default(self.kernel,
                                                      self.kernel.shape)
            biases = \
                tf.compat.v1.placeholder_with_default(self.biases,
                                                      self.biases.shape)
        else:
            raise ValueError("Invalid number of inputs for convolution"
                             " operator constructor: {}. Number of inputs"
                             " should be 1 or 3.".format(len(inpts)))

        # Inputs can be int8 but tf conv2d only supports floating point types
        # TODO: np vs tf
        if inpt.dtype not in ['float32', tf.float32]:
            inpt = tf.cast(inpt, RtLayerTF.dtype_to_tf[self.dtype])
        if kernel.dtype not in ['float32', tf.float32]:
            kernel = kernel.astype(RtLayerTF.dtype_to_np[self.dtype]) if \
                isinstance(kernel, np.ndarray) else \
                tf.cast(kernel, RtLayerTF.dtype_to_tf[self.dtype])

        kernel_layout, paddings, strides, dilations = \
            self.kernel_layout, self.paddings, self.strides, self.dilations

        if kernel_layout == 'OIHW' and self.kernel_groups > 1:
            # OIHW -> HWOI
            # NOTE: discrepancy between TVM and Tensorflow??
            kernel_trans = np.transpose(kernel, (2, 3, 0, 1)) if\
                isinstance(kernel, np.ndarray) else \
                tf.transpose(kernel, (2, 3, 0, 1))
            logger.debug("Kernel transposed shape: {}"
                         .format(kernel_trans.shape))
        elif kernel_layout == 'OHWI' and self.kernel_groups > 1:
            # OHWI -> HWOI
            # NOTE: discrepancy between TVM and Tensorflow??
            kernel_trans = np.transpose(kernel, (1, 2, 0, 3)) if\
                isinstance(kernel, np.ndarray) else \
                tf.transpose(kernel, (1, 2, 0, 3))
            logger.debug("Kernel transposed shape: {}"
                         .format(kernel_trans.shape))
        elif kernel_layout == 'OIHW':
            # OIHW -> HWIO
            # if isinstance(kernel, np.ndarray):
            #     kernel_trans = np.transpose(kernel, (2, 3, 1, 0))
            # elif isinstance(kernel, tf.constant):
            #     kernel_trans = np.transpose()
            # else:
            #     kernel_trans = tf.transpose(kernel, (2, 3, 1, 0))

            kernel_trans = np.transpose(kernel, (2, 3, 1, 0)) if\
                isinstance(kernel, np.ndarray) else \
                tf.transpose(kernel, (2, 3, 1, 0))
            logger.debug("Kernel transposed shape: {}"
                         .format(kernel_trans.shape))
        elif kernel_layout == 'OHWI':
            # OHWI -> HWIO
            # if isinstance(kernel, np.ndarray):
            #     kernel_trans = np.transpose(kernel, (1, 2, 3, 0))
            # elif isinstance(kernel, tf.constant):
            #     kernel_trans = np.transpose()
            # else:
            #     kernel_trans = tf.transpose(kernel, (1, 2, 3, 0))

            kernel_trans = np.transpose(kernel, (1, 2, 3, 0)) if\
                isinstance(kernel, np.ndarray) else \
                tf.transpose(kernel, (1, 2, 3, 0))
            logger.debug("Kernel transposed shape: {}"
                         .format(kernel_trans.shape))
        else:
            kernel_trans = kernel

        if self.layout == 'NCHW':
            paddings = [paddings[0], paddings[2], paddings[3], paddings[1]]
            strides = [strides[0], strides[2], strides[3], strides[1]]
            dilations = [dilations[0], dilations[2],
                         dilations[3], dilations[1]]

        if self.layout == 'NCHW':
            trans_inpt = tf.transpose(inpt, (0, 2, 3, 1))
            logger.debug("Input shape transformed: {}"
                         .format(trans_inpt.shape))
        else:
            trans_inpt = inpt

        # pad_along_height = max((out_height - 1) * strides[1] +
        #   filter_height - in_height, 0)
        out_h = self.shape[2] if self.layout == 'NCHW' else self.shape[1]
        out_w = self.shape[3] if self.layout == 'NCHW' else self.shape[2]
        in_h, strides_h, kernel_h = \
            int(trans_inpt.shape[1]), strides[1], int(kernel_trans.shape[0])
        in_w, strides_w, kernel_w = \
            int(trans_inpt.shape[2]), strides[2], int(kernel_trans.shape[1])
        # pad_along_height = max((out_h - 1) * strides[1] +
        #                        int(kernel_trans.shape[0]) -
        #                        int(trans_inpt.shape[1]), 0)
        # pad_along_width = max((out_w - 1) * strides[2] +
        #                       int(kernel_trans.shape[1]) -
        #                       int(trans_inpt.shape[2]), 0)
        logger.debug("Paddings: {}".format(paddings))
        logger.debug("out_h: {}, in_h: {}, out_w: {}, in_w: {}"
                     .format(out_h, in_h, out_w, in_w))

        if [list(pad) for pad in paddings] == [[0, 0], [0, 0], [0, 0], [0, 0]]:
            padded_inpt = trans_inpt
            padding_type = 'VALID'
        elif paddings[1][0] != paddings[1][1] and \
                paddings[2][0] != paddings[2][1] and \
                int(math.ceil(in_h / float(strides_h))) == out_h and \
                int(math.ceil(in_w / float(strides_w))) == out_w:
            # Asymmetric padding and Tensorflow SAME conditions -> SAME
            #   This is caused by TVM conversion of SAME padding and dpuv1
            #   compiler issue with asymmetric padding
            padding_type = 'SAME'
            padded_inpt = trans_inpt
            logger.debug("Padding type: SAME")
        else:
            padded_inpt = tf.pad(trans_inpt, paddings=paddings,
                                 mode="CONSTANT")
            padding_type = 'VALID'
            logger.debug("Padded input shape: {}".format(padded_inpt.shape))

        use_bias = (not isinstance(biases, np.ndarray)) or biases.any()
        if self.kernel_groups == 1:
            conv_res = tf.nn.conv2d(
                padded_inpt,
                kernel_trans,
                strides,
                padding_type,
                data_format='NHWC',
                dilations=dilations,
                name=self.name  # if not use_bias else self.name + '_Conv'
            )
        else:
            conv_res = tf.nn.depthwise_conv2d(
                input=padded_inpt,
                filter=kernel_trans,
                strides=strides,
                padding=padding_type
            )

        return conv_res

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        # assert(len(inpts) == 1)
        res = self._get_conv_tensor(inpts)

        if len(inpts) == 3:
            biases = inpts[2]
        elif len(inpts) == 1:
            biases = \
                tf.compat.v1.placeholder_with_default(self.biases,
                                                      self.biases.shape)
        else:
            raise ValueError("Invalid number of inputs for convolution"
                             " operator constructor: {}. Number of inputs"
                             " should be 1 or 3.".format(len(inpts)))

        if biases.dtype not in ['float32', tf.float32]:
            biases = tf.cast(biases, RtLayerTF.dtype_to_tf[self.dtype])

        if (not isinstance(biases, np.ndarray)) or biases.any():
            # Remove biases if numpy.ndarray and all zeros
            res = tf.nn.bias_add(res, biases, data_format='NHWC',
                                 name=self.name + '_Bias')
        # if self.use_activation not
        # in ['relu', 'leaky_relu']
        # else self.name + '_Bias')

        # TODO:name of convolution
        # ACTIVATIONS
        if self.use_activation == 'relu':
            res = tf.nn.relu(res, name=self.name + "_Relu")
        elif self.use_activation == 'leaky_relu':
            res = tf.nn.leaky_relu(
                res,
                alpha=self.activation_attrs['alpha'],
                name=self.name + "_Relu")

        if self.layout == 'NCHW':
            res = tf.transpose(res, (0, 3, 1, 2))

        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert(len(inputs) == len(self.input_shapes))
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)

    def get_output_for_quantization(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        """
        TODO
        """
        assert(len(inputs) == len(self.input_shapes))
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            # return sess.run(self.quant_output, feed_dict=feed_dict)
            return sess.run(self.res, feed_dict=feed_dict)


@rt_register_xlayer_2_tf_factory_func('Convolution')
def conv2d_factory():
    return base.get_conv2d_layer(ConvLayer, ConstantLayer)


###################
# Conv2DTranspose #
###################

class TensorInitializer(tf.keras.initializers.Initializer):

    def __init__(self, value, dtype='float32', verify_shape=False):

        self.value = tf.constant(value) if isinstance(value, np.ndarray) \
            else value

    def __call__(self, shape, dtype=None, partition_info=None):
        if shape != self.value.shape:
            raise ValueError("Incompatible shapes: {} and {}"
                             .format(shape, self.value.shape))

        if dtype is not None and dtype != self.value.dtype:
            self.value = tf.cast(self.value, dtype)

        return self.value


class Conv2DTransposeLayer(rt_layer.Conv2DTransposeLayer, RtLayerTF):

    def init(self):
        # () -> None
        """
        Initialize a transposed convolution layer on top of
            tf.nn.conv2d_tranpose operation
        """

        logger.info("Init transposed convolution layer")
        logger.debug(self.input_shapes)

        self.data_layout = self.attrs['data_layout']

        input_shapes, kernel, biases = \
            self.input_shapes, self.kernel, self.biases
        if kernel is not None:
            kernel = \
                tf.compat.v1.placeholder_with_default(kernel, kernel.shape)
        else:
            kernel = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[1])

        if biases is not None:
            biases = \
                tf.compat.v1.placeholder_with_default(biases, biases.shape)
        else:
            biases = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=input_shapes[2])

        inpt = tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                        shape=input_shapes[0])

        self.inpts = [inpt, kernel, biases]

        self.quant_output = self._get_conv_tensor(self.inpts, placeholder=True)
        self.res = self.get_output_tensors(self.inpts, placeholder=True)[0]

        logger.info("Res shape: {}".format(self.res.shape))

    def _get_conv_tensor(self, inpts, placeholder=False, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor

        if len(inpts) == 3:
            inpt, kernel, _ = inpts
        elif len(inpts) == 1:
            inpt = inpts[0]
            kernel = \
                tf.compat.v1.placeholder_with_default(self.kernel,
                                                      self.kernel.shape)
            # biases = tf.compat.v1.placeholder_with_default(self.biases,
            # self.biases.shape)
        else:
            raise ValueError("Invalid number of inputs for convolution"
                             " operator constructor: {}. Number of inputs"
                             " should be 1 or 3.".format(len(inpts)))

        # Inputs can be int8 but tf conv2d only supports floating point types
        inpt = tf.cast(inpt, RtLayerTF.dtype_to_tf[self.dtype])

        if isinstance(kernel, np.ndarray) and \
                kernel.dtype not in \
                [self.dtype, RtLayerTF.dtype_to_np[self.dtype]]:
            kernel = kernel.astype(RtLayerTF.dtype_to_np[self.dtype])
        elif kernel.dtype not in \
                [self.dtype, RtLayerTF.dtype_to_tf[self.dtype]]:
            kernel = tf.cast(inpt, RtLayerTF.dtype_to_tf[self.dtype])

        kernel_layout, paddings, strides, dilations = \
            self.kernel_layout, self.paddings, self.strides, self.dilations

        if kernel_layout == 'OIHW':
            # OIHW -> HWOI
            kernel_trans = np.transpose(kernel, (2, 3, 0, 1)) if\
                isinstance(kernel, np.ndarray) else \
                tf.transpose(kernel, (2, 3, 0, 1))
            logger.debug("Kernel transposed shape: {}"
                         .format(kernel_trans.shape))
        elif kernel_layout == 'OHWI':
            # OHWI -> HWOI
            kernel_trans = np.transpose(kernel, (1, 2, 0, 3)) if\
                isinstance(kernel, np.ndarray) else \
                tf.transpose(kernel, (1, 2, 0, 3))
            logger.debug("Kernel transposed shape: {}"
                         .format(kernel_trans.shape))
        else:
            kernel_trans = kernel

        if self.data_layout == 'NCHW':
            paddings = [paddings[0], paddings[2], paddings[3], paddings[1]]
            strides = [strides[0], strides[2], strides[3], strides[1]]
            dilations = [dilations[0], dilations[2], dilations[3],
                         dilations[1]]

        if self.data_layout == 'NCHW':
            # NCHW -> NHWC
            trans_inpt = tf.transpose(inpt, (0, 2, 3, 1))
            logger.debug("Input shape transformed: {}"
                         .format(trans_inpt.shape))

            # TODO this is more convenient in Tensorflow 1.14
            # NCHW -> NHWC
            output_shape_hwc = \
                [self.shape[2], self.shape[3], self.shape[1]]
        else:
            trans_inpt = inpt
            output_shape_hwc = self.shape[1:].tolist()

        # !! TODO: Padding handled differently in different frameworks
        #   e.g. PyTorch and Tensorflow, see (https://stackoverflow.com/
        #   questions/51146217/tensorflow-conv2d-transpose-size-of-out-
        #   backprop-doesnt-match-computed)

        kernel_size = [int(kernel_trans.shape[0]), int(kernel_trans.shape[1])]
        strides_hw = [strides[1], strides[2]]

        # NHWC format here
        assert paddings[1][0] == paddings[1][1]
        assert paddings[2][0] == paddings[2][1]
        padding_hw = [paddings[1][0], paddings[2][0]]
        if padding_hw[0] == (kernel_size[0] - strides_hw[0]) // 2 and\
                padding_hw[1] == (kernel_size[1] - strides_hw[1]) // 2:
            padding_type = 'SAME'
        elif padding_hw[0] == 0 and padding_hw[1] == 0:
            padding_type = 'VALID'
        else:
            pad_same_h = (kernel_size[0] - strides_hw[0]) / 2
            pad_same_w = (kernel_size[1] - strides_hw[1]) / 2
            raise NotImplementedError("Unsupported padding for Conv2DTranspose"
                                      " Only Tensorflow padding 'SAME' and"
                                      " 'VALID' are supported but got: {}"
                                      " which does not translate to 'SAME'"
                                      " == [{}, {}] or 'VALID'== [0, 0]"
                                      .format(padding_hw, pad_same_h,
                                              pad_same_w))

        # TODO this is more convenient in Tensorflow 1.14,
        # NOTE This dynamic shape retrieval creates complex tensorflow models,
        #   for now we only support batch size 1
        if self.batch_size == -1:
            in_batch_sz = tf.shape(trans_inpt)[0:1]
            output_shape = tf.concat([in_batch_sz, output_shape_hwc], axis=0)
        else:
            output_shape = tf.stack([self.batch_size] + output_shape_hwc)

        if self.kernel_groups == 1:
            if self.placeholder is True:
                # For stepwise tensorflow model, TODO: remove stepwise
                conv_res = tf.nn.conv2d_transpose(
                    trans_inpt,
                    kernel_trans,
                    strides=strides,
                    padding=padding_type,
                    output_shape=output_shape,
                    name=self.name
                    # dilations=dilations TODO: not available in 1.13
                )
            else:
                channels = output_shape_hwc[2]
                input_shape = tf.shape(trans_inpt)
                batch_size = input_shape[0]
                in_h = input_shape[1]
                in_w = input_shape[2]

                def out_size(in_size, stride, padding_type, kernel_size):
                    # See Tensorflow https://github.com/tensorflow/tensorflow/blob/5b900cfe4b3b848f577315a0dde09a729f770e95/tensorflow/python/keras/utils/conv_utils.py#L140 # noqa
                    in_size *= stride
                    if padding_type == 'VALID':
                        in_size += max(kernel_size - stride, 0)
                    # elif padding == 'FULL':
                    #     in_size-= (stride + filter_size - 2)
                    return in_size

                out_h = \
                    out_size(in_h, strides_hw[0], padding_type, kernel_size[0])
                out_w = \
                    out_size(in_w, strides_hw[1], padding_type, kernel_size[1])

                output_shape_reconst = \
                    tf.stack((batch_size, out_h, out_w, channels))

                conv_res = tf.nn.conv2d_transpose(
                    trans_inpt,
                    kernel_trans,
                    strides=strides,
                    padding=padding_type,
                    output_shape=output_shape_reconst,
                    name=self.name
                    # dilations=dilations TODO: not available in 1.13
                )

                # conv = tf.layers.Conv2DTranspose(
                #     output_shape_hwc[2],
                #     kernel_size,
                #     strides[1:3],
                #     padding=padding_type,
                #     use_bias=False,
                #     kernel_initializer=TensorInitializer(kernel_trans),
                #     name=self.name + "_Conv"  # + "_Conv"
                # )
                # # conv.kernel = kernel_trans
                # conv_res = tf.identity(conv(trans_inpt), name=self.name)
        else:
            raise NotImplementedError("Transposed depthwise convolution not"
                                      " implemented")

        return conv_res

    def get_output_tensors(self, inpts, placeholder=False, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        # assert(len(inpts) == 1)
        res = self._get_conv_tensor(inpts, placeholder=placeholder)

        if len(inpts) == 3:
            biases = inpts[2]
        elif len(inpts) == 1:
            biases = \
                tf.compat.v1.placeholder_with_default(self.biases,
                                                      self.biases.shape)
        else:
            raise ValueError("Invalid number of inputs for convolution"
                             " operator constructor: {}. Number of inputs"
                             " should be 1 or 3.".format(len(inpts)))

        # biases = tf.cast(biases, RtLayerTF.dtype_to_tf[self.dtype])
        # res = tf.add(res, biases)
        if biases.dtype not in ['float32', tf.float32]:
            biases = tf.cast(biases, RtLayerTF.dtype_to_tf[self.dtype])

        if (not isinstance(biases, np.ndarray)) or biases.any():
            # Remove biases if numpy.ndarray and all zeros
            res = tf.nn.bias_add(res, biases, data_format='NHWC',
                                 name=self.name + '_Bias')
            # if self.use_activation not
            # in ['relu', 'leaky_relu']
            # else self.name + '_Bias')

        if self.use_activation == 'relu':
            res = tf.nn.relu(res, name=self.name + '_Relu')
        elif self.use_activation == 'leaky_relu':
            res = tf.nn.leaky_relu(
                res,
                alpha=self.activation_attrs['alpha'],
                name=self.name + '_Relu'
            )

        if self.data_layout == 'NCHW':
            res = tf.transpose(res, (0, 3, 1, 2))

        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        assert len(inputs) == len(self.input_shapes)
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)

    def get_output_for_quantization(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray
        """
        TODO
        """
        assert(len(inputs) == len(self.input_shapes))
        feed_dict = {
            self.inpts[i]: inputs[i] for i in range(len(inputs))
        }

        with tf.compat.v1.Session() as sess:
            # return sess.run(self.quant_output, feed_dict=feed_dict)
            return sess.run(self.res, feed_dict=feed_dict)


@rt_register_xlayer_2_tf_factory_func('Conv2DTranspose')
def conv2d_transpose_factory():
    return base.get_conv2d_transpose_layer(Conv2DTransposeLayer, ConstantLayer)

###########
# Flatten #
###########


@rt_register_xlayer_2_tf('Flatten')
class FlattenLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Input shape: {}".format(self.inpt.shape))
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        return [tf.contrib.layers.flatten(inpts[0])]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})

###########
# Pooling #
###########


class PoolingLayer(rt_layer.PoolingLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.layout = self.attrs['data_layout']

        logger.info("Init pooling layer: {}, shape: {}".format(self.op,
                                                               self.shape))
        logger.debug("Paddings: {}".format(self.paddings))

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        op, ksize, paddings, strides = \
            self.op, self.ksize, self.paddings, self.strides

        if self.layout == 'NCHW':
            ksize = [ksize[0], ksize[2], ksize[3], ksize[1]]
            paddings = [paddings[0], paddings[2], paddings[3], paddings[1]]
            strides = [strides[0], strides[2], strides[3], strides[1]]

        if op == 'Max':
            tf_pool_func = tf.nn.max_pool
        elif op == 'Avg':
            tf_pool_func = tf.nn.avg_pool2d
        else:
            raise NotImplementedError("Provided pooling operation is not"
                                      " supported at this moment: {}"
                                      .format(op))

        inpt = tf.cast(inpts[0], RtLayerTF.dtype_to_tf[self.dtype])
        if self.layout == 'NCHW':
            inpt = tf.transpose(inpt, (0, 2, 3, 1))
            logger.debug("Input shape transformed: {}".format(inpt.shape))

        # pad_along_height = max((out_height - 1) * strides[1] +
        #   filter_height - in_height, 0)
        out_h = self.shape[2] if self.layout == 'NCHW' else self.shape[1]
        out_w = self.shape[3] if self.layout == 'NCHW' else self.shape[2]
        in_h, strides_h = int(inpt.shape[1]), strides[1]
        in_w, strides_w = int(inpt.shape[2]), strides[2]
        # pad_along_height = max((out_h - 1) * strides[1] + ksize[1] -
        #                        int(inpt.shape[1]), 0)
        # pad_along_width = max((out_w - 1) * strides[2] + ksize[2] -
        #                       int(inpt.shape[2]), 0)
        # logger.debug("Paddings: {}".format(paddings))
        # logger.debug("out_h: {}, pad_along_h: {}, out_w: {}, pad_along_w: {}"
        #             .format(out_h, pad_along_height, out_w, pad_along_width))

        if [list(pad) for pad in paddings] == [[0, 0], [0, 0], [0, 0], [0, 0]]:
            padded_inpt = inpt
            padding_type = 'VALID'
        elif paddings[1][0] != paddings[1][1] and \
                paddings[2][0] != paddings[2][1] and \
                int(math.ceil(in_h / float(strides_h))) == out_h and\
                int(math.ceil(in_w / float(strides_w))) == out_w:
            padding_type = 'SAME'
            padded_inpt = inpt
            logger.debug("Padding type: SAME")
        else:
            padded_inpt = tf.pad(inpt, paddings=paddings, mode="CONSTANT")
            logger.debug("Padded input shape: {}".format(padded_inpt.shape))
            padding_type = 'VALID'

        res = tf_pool_func(
            padded_inpt,
            ksize,
            strides,
            padding_type,
            data_format='NHWC',
            name=self.name
        )

        if self.layout == 'NCHW':
            res = tf.transpose(res, (0, 3, 1, 2))

        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('Pooling')
def pooling_factory():
    return base.get_pooling_layer(PoolingLayer)


class PoolingNoDivisionLayer(PoolingLayer):

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        """
        NOTE: On FPGA, average pooling is computed as a sum over the
            respective tiles
            without division to get the average. The division is left to the
            quantization parameters. Therefore, we multiply the avgpool result
            again by the divisor for FPGA hardware simulation
        """
        res = super(PoolingNoDivisionLayer, self).get_output_tensors(inpts)[0]

        if self.op == 'Avg':
            avg_pool_divisor = np.prod(self.ksize)
            res = tf.multiply(res, avg_pool_divisor)
            # res = tf.round(res)

        return [res]


@rt_register_xlayer_2_tf_factory_func('PoolingNoDivision')
def pooling_no_division_factory():
    return base.get_pooling_layer(PoolingNoDivisionLayer)


################
# Upsampling2D #
################

@rt_register_xlayer_2_tf('Upsampling2D')
class Upsampling2DLayer(rt_layer.BaseLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        self.scale_h = self.attrs['scale_h']
        self.scale_w = self.attrs['scale_w']
        self.layout = self.attrs['data_layout']
        self.method = self.attrs['method']
        self.align_corners = self.attrs['align_corners']

        # if self.align_corners is True:
        #     raise ValueError("Tensorflow runtime only supports Upsampling2D"
        #                      " with align_corners set to False")
        if self.method not in ['nearest_neighbor', 'bilinear', 'bicubic']:
            raise ValueError("Tensorflow runtime only supports Upsampling2D"
                             " with nearest_neighbor, bilinear or bicubic"
                             " method, but got: {}"
                             .format(self.method))
        if self.layout not in ['NCHW', 'NHWC']:
            raise ValueError("Tensorflow runtime only supports Upsampling2D"
                             " with layout of 'NCHW' or 'NHWC' but got: {}"
                             .format(self.layout))

        # self.data_format = \
        #     {'NCHW': 'channels_first', 'NHWC': 'channels_last'}[self.layout]
        # self.interpolation = {
        #     'nearest_neighbor': 'nearest',
        #     'bilinear': 'bilinear',}[self.method]

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Input shape: {}".format(self.inpt.shape))
        logger.info("Output shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert len(inpts) == 1

        res = inpts[0]
        if self.layout == 'NCHW':
            h, w = self.input_shapes[0][2], self.input_shapes[0][3]
            res = tf.transpose(res, (0, 2, 3, 1))
        else:
            h, w = self.input_shapes[0][1], self.input_shapes[0][2]

        new_h, new_w = int(h * self.scale_h), int(w * self.scale_w)

        if self.method == 'nearest_neighbor':
            res = tf.compat.v1.image.resize_nearest_neighbor(
                res,
                size=[new_h, new_w],
                align_corners=self.align_corners,
                name=self.name
            )
        elif self.method == 'bilinear':
            res = tf.image.resize_bilinear(
                res,
                size=[new_h, new_w],
                align_corners=self.align_corners,
                name=self.name
            )
        elif self.method == 'bicubic':
            res = tf.image.resize_bicubic(
                res,
                size=[new_h, new_w],
                align_corners=self.align_corners,
                name=self.name
            )

        if self.layout == 'NCHW':
            res = tf.transpose(res, (0, 3, 1, 2))

        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})
