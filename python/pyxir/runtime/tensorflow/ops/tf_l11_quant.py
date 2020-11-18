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

#################
# QuantizeLayer #
#################


class QuantizeLayer(rt_layer.QuantizeLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp
        """
        logger.info("Quantizelayer init")
        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.input_types[0]],
            shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)

        threshold, axis, bitwidth, do_rounding = \
            self.threshold, self.axis, self.bitwidth, self.do_rounding
        threshold_shape = [1]*len(self.input_shapes[0])
        threshold_shape[axis] = len(threshold)
        threshold = np.reshape(np.array(threshold), threshold_shape)

        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1
        quant_factor = half_range / threshold

        val = tf.maximum(tf.minimum(inpts[0], threshold), -threshold)
        val = tf.multiply(val, quant_factor)

        if do_rounding:
            val = tf.round(val)

        res = tf.cast(val, RtLayerTF.dtype_to_tf[self.dtype])
        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('Quantize')
def quantize_factory():
    return base.get_quantize_layer(QuantizeLayer)

###################
# UnQuantizeLayer #
###################


class UnQuantizeLayer(rt_layer.UnQuantizeLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp
        """

        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.input_types[0]],
            shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        threshold, axis, bitwidth = self.threshold, self.axis, self.bitwidth

        threshold_shape = [1]*len(self.input_shapes[0])
        threshold_shape[axis] = len(threshold)
        threshold = np.reshape(np.array(threshold), threshold_shape)

        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1
        un_quant_factor = threshold / half_range

        res = tf.cast(inpts[0], RtLayerTF.dtype_to_tf[self.dtype])
        res = tf.multiply(res, un_quant_factor)

        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('UnQuantize')
def unquantize_factory():
    return base.get_unquantize_layer(UnQuantizeLayer)

################
# QuantizeBias #
################


class QuantizeBiasLayer(rt_layer.QuantizeBiasLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp, XDNNV3QuantizeBias
        """
        logger.info("Init quantize bias layer")

        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.input_types[0]],
            shape=self.input_shapes[0])
        # logger.debug("Threshold_bias shape: {}".format(threshold_bias.shape))

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor

        threshold_bias, threshold_ext, bitwidth, do_rounding = \
            self.threshold_bias, self.threshold_ext, self.bitwidth,\
            self.do_rounding

        logger.debug("Threshold_ext: {}".format(threshold_ext))
        # logger.debug("Threshold_bias: {}".format(threshold_bias))
        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1
        sf_ext = threshold_ext / half_range
        sf_bias = threshold_bias / half_range
        sf_acc = sf_ext * sf_bias
        # logger.debug("sf_acc: {}".format(sf_acc))

        # TODO
        if sf_acc.shape[0] != inpts[0].shape[0]:
            assert(inpts[0].shape[0] % sf_acc.shape[0] == 0)
            logger.warn("[WARNING] Adjusting threshold shape from: {} to {}"
                        " for quantize bias layer."
                        .format(sf_acc.shape[0], inpts[0].shape[0]))
            sf_acc = np.tile(sf_acc, inpts[0].shape[0] // sf_acc.shape[0])

        # 8 bit only for now
        if bitwidth == 8:
            macc_range = np.power(2, 24)
            macc_half_range = (macc_range / 2) - 1
            th_acc = sf_acc * macc_half_range
        # logger.debug("th_acc: {}".format(th_acc))

        val = tf.maximum(tf.minimum(inpts[0], th_acc), -th_acc)
        val = tf.divide(val, sf_acc)

        if do_rounding:
            val = tf.round(val)

        res = tf.cast(val, RtLayerTF.dtype_to_tf[self.dtype])
        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('QuantizeBias')
def quantize_bias_factory():
    return base.get_quantize_bias_layer(QuantizeBiasLayer)

#####################
# QuantizeScaleBias #
#####################


class QuantizeScaleBiasLayer(rt_layer.QuantizeScaleBiasLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp, XDNNQuantizeBiasV3_scale
        # TODO discrepancy with xdnn.cpp implementation
        """
        logger.info("Init quantize scale bias layer")

        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.input_types[0]],
            shape=self.input_shapes[0])
        # logger.debug("Threshold_bias shape: {}".format(threshold_bias.shape))

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        """
        Get output tensor for quantizing bias layer in scaling operation

        bias / (th_scale * th_in / 127) *(th_in / 127 * 127 / th_out * gamma)
            = bias * gamma/th_scale * 127 / th_out
            +-= bias * 127/th_out
        Therefore we divide bias by (th_scale * th_in / 127) in this
        quantization layer
        """
        assert(len(inpts) == 1)

        inpt = inpts[0]
        logger.debug(inpt.dtype)
        # th_scale, th_ext, bitwidth, do_rounding = \
        #    self.th_scale, self.th_ext, self.bitwidth, self.do_rounding
        scale, postscale_shift, th_out, bitwidth, do_rounding = \
            self.scale, self.postscale_shift, self.th_out, self.bitwidth,\
            self.do_rounding

        # logger.debug("Threshold_ext: {}".format(th_ext))
        # logger.debug("Threshold_bias: {}".format(threshold_bias))
        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1
        # sf_ext = th_ext / half_range
        # sf_scale = th_scale / half_range
        sf_out = th_out / half_range
        # ! np.absolute for negative scale
        sf_acc = np.squeeze(
            sf_out * np.multiply(
                np.absolute(scale),
                np.power(2, -np.array(postscale_shift).astype(np.float32)))
        )
        # sf_acc = tf.constant(np.squeeze(sf_acc), dtype=tf.float32)
        # logger.debug("sf_acc: {}".format(sf_acc))
        # TODO

        # sf_acc = sf_ext * th_scale
        # logger.debug("sf_acc: {}".format(sf_acc))

        # TODO
        if sf_acc.shape[0] != inpts[0].shape[0]:
            assert(inpts[0].shape[0] % sf_acc.shape[0] == 0)
            logger.warn("[WARNING] Adjusting threshold shape from: {} to {}"
                        " for quantize scale bias layer."
                        .format(sf_acc.shape[0],
                                inpts[0].shape[0]))
            sf_acc = np.tile(sf_acc, inpts[0].shape[0] // sf_acc.shape[0])

        # 8 bit only for now
        if bitwidth == 8:
            macc_range = np.power(2, 32)
            macc_half_range = (macc_range / 2) - 1
            th_acc = sf_acc * macc_half_range
            logger.debug("th_acc: {}".format(type(th_acc)))
            logger.debug("th_acc: {}".format(th_acc.dtype))
            th_acc = tf.constant(th_acc, dtype=tf.float32)
        logger.debug("th_acc: {}".format(th_acc.dtype))

        logger.debug(inpt.dtype)
        val0 = tf.maximum(tf.minimum(inpt, th_acc), -th_acc)

        # Robust division
        scale_division_robust = tf.constant(
            half_range / th_out *
            np.power(2, np.array(postscale_shift).astype(np.float32)) /
            scale,
            dtype=tf.float32)  # TODO avoid scale of 0 division
        # val = tf.divide(val, sf_acc)

        val = tf.multiply(val0, scale_division_robust)

        if do_rounding:
            val = tf.round(val)

        res = tf.cast(val, RtLayerTF.dtype_to_tf[self.dtype])
        return [res]  # , val0, val1, val, scale_division_robust, inpt]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('QuantizeScaleBias')
def quantize_scale_bias_factory():
    return base.get_quantize_scale_bias_layer(QuantizeScaleBiasLayer)

#################
# QuantizeInter #
#################


class QuantizeInterLayer(rt_layer.QuantizeInterLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp, XDNNV3QuantizeInterLayer
        """
        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf['int64'],  # TODO
            shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        input_shape, prescale_shift, scale, postscale_shift, axis, bitwidth = \
            self.input_shapes[0], self.prescale_shift, self.scale, \
            self.postscale_shift, self.axis, self.bitwidth

        scale_shape = [1] * len(input_shape)
        scale_shape[axis] = len(scale)
        scale = np.array(scale).reshape(scale_shape)
        postscale_shift = np.array(postscale_shift).reshape(scale_shape)
        # logger.debug("scale: {}".format(scale))
        # logger.debug("postscale_shift: {}".format(postscale_shift))

        # Quantize bias for depthwise convolution
        if scale.shape[axis] != inpts[0].shape[axis]:
            assert(inpts[0].shape[axis] % scale.shape[axis] == 0)
            logger.warn("[WARNING] Adjusting threshold shape from: {} to {}"
                        "for quantize inter layer."
                        .format(scale.shape[axis], inpts[0].shape[axis]))
            tile_shape = [1] * len(input_shape)
            tile_shape[axis] = inpts[0].shape[axis] // scale.shape[axis]
            scale = np.tile(scale, tile_shape)
            postscale_shift = np.tile(postscale_shift, tile_shape)

        pre_max_range = np.power(2, 24)  # MACC to 24 bits on HW
        pre_max_val = (pre_max_range / 2) - 1  # TODO, use this to clip acc
        pre_min_val = -(pre_max_range / 2)

        post_max_range = np.power(2, bitwidth)
        post_max_val = (post_max_range / 2) - 1
        post_min_val = -(post_max_range / 2)

        inpt = tf.cast(inpts[0], RtLayerTF.dtype_to_tf['int64'])
        scale = tf.cast(scale, RtLayerTF.dtype_to_tf['int64'])
        postscale_shift_tf = tf.cast(postscale_shift,
                                     RtLayerTF.dtype_to_tf['int64'])

        zeros = tf.zeros_like(postscale_shift_tf, dtype=tf.int64)

        res = tf.multiply(inpt, scale)
        # self.test = res
        logger.debug("scale: {}".format(scale.shape))
        logger.debug("postscale_shift: {}".format(postscale_shift.shape))
        logger.debug("post_max_val: {}".format(post_max_val))
        logger.debug("post_min_val: {}".format(post_min_val))
        if np.all(postscale_shift >= 0):
            res = tf.bitwise.right_shift(
                res,
                tf.maximum(zeros, tf.subtract(postscale_shift_tf, 1)))
            res = tf.add(res, 1)
            res = tf.bitwise.right_shift(res, 1)
        else:
            raise ValueError("Postscale shift should not be less than 0 but"
                             " was: {}. This will give in wrong results down"
                             " the road.".format(postscale_shift))

        # TODO: compensate for negative scaling (15>>1 = 7, -15>>1 = -8)
        ####
        # ones = tf.ones(scale.shape, dtype=RtLayerTF.dtype_to_tf['int64'])
        # zeros = tf.zeros(scale.shape, dtype=RtLayerTF.dtype_to_tf['int64'])
        # minus_one = tf.constant(-1, dtype=RtLayerTF.dtype_to_tf['int64'])
        # negative_scaling_adjustment = tf.where(
        #     tf.equal(tf.sign(scale), minus_one), ones, zeros
        # )
        # res = tf.add(res, negative_scaling_adjustment)
        ####

        res = tf.maximum(tf.minimum(res, post_max_val), post_min_val)

        # ! For scaling layer with negative scaling values merged with relu,
        #   we have to take the relu into account after quantization scaling
        #   instead of before
        # TODO
        if self.relu:
            res = tf.nn.relu(res)

        res = tf.cast(res, RtLayerTF.dtype_to_tf[self.dtype])
        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            # logger.debug("TEST")
            # logger.debug(sess.run(self.test,
            #              feed_dict={self.inpt: inputs[0]}))
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('QuantizeInter')
def quantize_inter_factory():
    return base.get_quantize_inter_layer(QuantizeInterLayer)


###############################
# QuantizeInter12MostSignBits #
###############################

class QuantizeInter12MSBitsLayer(rt_layer.QuantizeInterLayer, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp, XDNNV3QuantizeInterLayer
        """
        logger.info("Init QuantizeInter12MSBitsLayer: {}".format(self.name))
        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf['int64'],  # TODO
            shape=self.input_shapes[0])

        self.res = self.get_output_tensors([self.inpt])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts, **kwargs):
        # type: (List[tf.Tensor]) -> tf.Tensor
        assert(len(inpts) == 1)
        input_shape, prescale_shift, scale, postscale_shift, axis, bitwidth = \
            self.input_shapes[0], self.prescale_shift, self.scale, \
            self.postscale_shift, self.axis, self.bitwidth

        scale_shape = [1] * len(input_shape)
        scale_shape[axis] = len(scale)
        scale = np.array(scale).reshape(scale_shape)
        postscale_shift = np.array(postscale_shift).reshape(scale_shape)
        # logger.debug("scale: {}".format(scale))
        # logger.debug("postscale_shift: {}".format(postscale_shift))

        # Quantize bias for depthwise convolution
        if scale.shape[axis] != inpts[0].shape[axis]:
            assert(inpts[0].shape[axis] % scale.shape[axis] == 0)
            logger.warn("[WARNING] Adjusting threshold shape from: {} to {}"
                        "for quantize inter layer."
                        .format(scale.shape[axis], inpts[0].shape[axis]))
            tile_shape = [1] * len(input_shape)
            tile_shape[axis] = inpts[0].shape[axis] // scale.shape[axis]
            scale = np.tile(scale, tile_shape)
            postscale_shift = np.tile(postscale_shift, tile_shape)

        pre_max_range = np.power(2, 24)  # MACC to 24 bits on HW
        pre_max_val = (pre_max_range / 2) - 1  # TODO, use this to clip acc
        pre_min_val = -(pre_max_range / 2)

        post_max_range = np.power(2, bitwidth)
        post_max_val = (post_max_range / 2) - 1
        post_min_val = -(post_max_range / 2)

        inpt = tf.cast(inpts[0], RtLayerTF.dtype_to_tf['int64'])
        scale = tf.cast(scale, RtLayerTF.dtype_to_tf['int64'])
        postscale_shift_tf = tf.cast(postscale_shift,
                                     RtLayerTF.dtype_to_tf['int64'])

        zeros = tf.zeros_like(postscale_shift_tf, dtype=tf.int64)

        # ! Clip 12 most significant bits
        # res = tf.bitwise.right_shift(inpt, 4)
        # res = tf.bitwise.left_shift(res, 4)
        res = tf.bitwise.bitwise_and(inpt, ~0 << 4)

        res = tf.multiply(res, scale)
        # self.test = res
        logger.debug("scale: {}".format(scale.shape))
        logger.debug("postscale_shift: {}".format(postscale_shift.shape))
        logger.debug("post_max_val: {}".format(post_max_val))
        logger.debug("post_min_val: {}".format(post_min_val))
        if np.all(postscale_shift >= 0):
            res = tf.bitwise.right_shift(
                res,
                tf.maximum(zeros, tf.subtract(postscale_shift_tf, 1)))
            res = tf.add(res, 1)
            res = tf.bitwise.right_shift(res, 1)
        else:
            raise ValueError("Postscale shift should not be less than 0 but"
                             " was: {}. This will give in wrong results down"
                             " the road.".format(postscale_shift))

        # TODO: compensate for negative scaling (15>>1 = 7, -15>>1 = -8)
        ####
        # ones = tf.ones(scale.shape, dtype=RtLayerTF.dtype_to_tf['int64'])
        # zeros = tf.zeros(scale.shape, dtype=RtLayerTF.dtype_to_tf['int64'])
        # minus_one = tf.constant(-1, dtype=RtLayerTF.dtype_to_tf['int64'])
        # negative_scaling_adjustment = tf.where(
        #     tf.equal(tf.sign(scale), minus_one), ones, zeros
        # )
        # res = tf.add(res, negative_scaling_adjustment)
        ####

        res = tf.maximum(tf.minimum(res, post_max_val), post_min_val)

        # ! For scaling layer with negative scaling values merged with relu,
        #   we have to take the relu into account after quantization scaling
        #   instead of before
        # TODO
        if self.relu:
            res = tf.nn.relu(res)

        res = tf.cast(res, RtLayerTF.dtype_to_tf[self.dtype])
        return [res]

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        assert(len(inputs) == 1)

        with tf.compat.v1.Session() as sess:
            # logger.debug("TEST")
            # logger.debug(sess.run(self.test,
            #              feed_dict={self.inpt: inputs[0]}))
            return sess.run(self.res, feed_dict={self.inpt: inputs[0]})


@rt_register_xlayer_2_tf_factory_func('QuantizeInter12MSBits')
def quantize_inter_factory():
    return base.get_quantize_inter_layer(QuantizeInter12MSBitsLayer)
