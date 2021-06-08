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

"""Module for xf layers doing quantization using MSE threshold clipping approach"""

import abc
import numpy as np
import tensorflow as tf
import logging

from typing import List

from pyxir.shapes import TensorShape, TupleShape
from pyxir.runtime.rt_layer import RtLayer
from pyxir.runtime.tensorflow.rt_layer_tf import RtLayerTF
from pyxir.runtime.tensorflow.ops.tf_l1_basic_nn import ReluLayer
from pyxir.runtime.tensorflow.runtime_tf import X_2_TF

logger = logging.getLogger("pyxir")

# Suppress warnings
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def MSE(x1, x2):
    return ((x1 - x2)**2).mean()


# MSEQuantizeLayer

class MSEQuantizeLayerBase(RtLayer):

    """
    Generic ... TODO

    Arguments
    ---------
    name: str
        the name of this layer
    shape: List[int]/Tuple[int]
        the shape of this layer
    dtype: str
        the type of this layer
    inputs: List[str]
        the input names of this layer
    input_shapes: List[List[int]/Tuple[int]]
        the input shapes for all the inputs
    subgrap: str
        the subgraph to which the layer belongs
    axis: int
        specifies the axis on which the threshold should be applied
    bitwidth: int
        the bitwidth to be quantized to
    do_rounding: bool
        whether to round instead of cast floats to ints
    """

    def __init__(self,
                 name,
                 shape,
                 dtype,
                 inputs,
                 input_shapes,
                 subgraph,
                 axis,
                 bitwidth,
                 do_rounding=True,
                 mse_opt_num=50):
        # TODO: checks
        super(MSEQuantizeLayerBase, self).__init__(name, 'MSEQuantize', shape,
                                                   dtype, inputs, input_shapes,
                                                   subgraph)

        # TODO check shape
        # Threshold is an input to the layer
        assert(len(self.input_shapes) in [2, 4])
        self.axis = axis
        self.bitwidth = bitwidth
        self.do_rounding = do_rounding
        self.mse_opt_num = mse_opt_num

        if self.bitwidth != 8:
            raise NotImplementedError("QuantizeForThresholdTraining layer only"
                                      " supports bitwith 8 for now")

        # INITIALIZE
        self.init()


class MSEQuantizeLayer(MSEQuantizeLayerBase, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See xfdnn/rt/xdnn_cpp/xdnn.cpp
        """
        logger.info("MSEQuantizelayer init, nb inputs: {}"
                    .format(len(self.input_shapes)))
        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.dtype],
            shape=self.input_shapes[0])
        self.th_params = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.dtype],
            shape=self.input_shapes[1])
        inpts = [self.inpt, self.th_params]

        # TODO
        if len(self.input_shapes) > 2:
            self.alt_inpt = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=self.input_shapes[2])
            self.beta = \
                tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                         shape=self.input_shapes[3])
            inpts.extend([self.alt_inpt, self.beta])

        logger.info("In shape: {}, th_params shape: {}"
                    .format(self.inpt.shape, self.th_params.shape))

        self.res = self.get_output_tensors(inpts)[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert(len(inpts) == len(self.input_shapes))

        # logger.debug([type(inpt) for inpt in inpts])
        if len(self.input_shapes) == 4:
            inpt, threshold, alt_inpt, beta = \
                inpts[0], inpts[1], inpts[2], inpts[3]
        else:
            inpt, threshold = inpts
            alt_inpt, beta = None, None

        axis, bitwidth, do_rounding = \
            self.axis, self.bitwidth, self.do_rounding
        mse_opt_num = self.mse_opt_num

        threshold_shape = [1]*len(self.input_shapes[0])
        threshold_shape[axis] = int(threshold.shape[0])
        logger.debug("threshold_shape: {}".format(threshold_shape))

        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1

        #
        def quantize_unquantize(x, th):
            # Quantize
            th = np.reshape(np.array(th), threshold_shape)
            quant_factor = half_range / th

            qx = np.maximum(np.minimum(x, th), -th)
            qx = qx * quant_factor

            if do_rounding:
                qx = np.round(qx)

            qx = np.floor(qx)  # to int

            # Unquantize
            un_quant_factor = th / half_range
            res = qx * un_quant_factor

            return res

        def mse_optimization(x, th):
            # logger.debug(x.shape, th.shape)
            # TODO #np.isscalar(th) or
            nonlocal mse_opt_num, beta

            axis = None if th.shape in [(), (1,)] else (1, 2, 3)
            # OIHW weights
            th_max = np.array(np.amax(np.absolute(x), axis=axis))
            th_max = np.expand_dims(th_max, axis=0) if th_max.shape == ()\
                else th_max
            # logger.debug(th.shape, th_max.shape)
            assert(th_max.shape == th.shape)  # or \
            # (th_max.shape in [(), (1,)] and th.shape in [(), (1,)]))

            mse_th = MSE(quantize_unquantize(x, th), x)
            mse_th_max = MSE(quantize_unquantize(x, th_max), x)
            # logger.debug(type(th), type(mse_th))
            # logger.debug(type(th_max), type(mse_th_max))
            if mse_th <= mse_th_max:
                best_th = th
                best_mse = mse_th
            else:
                best_th = th_max
                best_mse = mse_th_max

            for _ in range(0, mse_opt_num):
                ra = np.random.rand()  # np.random.rand(*th.shape)
                while ra == 0.:
                    ra = np.random.rand()
                # ra[ra == 0.] = ra.mean() + 0.0001 # replace zeros
                th_rand = ra * th_max
                # TODO TEST
                x_h = quantize_unquantize(x, th_rand)
                # if beta is not None:
                #     mean_axis = (0,2,3) if th.shape in [(), (1,)]
                #       else (1,2,3)
                #     #beta_est = (x - x_h).mean(axis=mean_axis)
                #     beta_est = np.median(x - x_h, axis=mean_axis)
                #     beta_est = np.expand_dims(beta_est, 1)
                #     beta_est = np.expand_dims(beta_est, 1)
                #     beta_est = np.expand_dims(beta_est, 0)
                #     x_h += beta_est

                mse_rand = MSE(x_h, x)
                if mse_rand < best_mse:
                    best_th, best_mse = th_rand, mse_rand

            return best_th.astype(np.float32)

        def beta_optimization(x, th):
            nonlocal beta, axis

            beta_est = None
            if beta is not None:
                x_best = quantize_unquantize(x, th)
                # TODO assuming NCHW and OIHW which we can deduct from th shape
                # TODO what if O == 1 in OIHW weight
                mean_axis = (0, 2, 3) if th.shape in [(), (1,)] else (1, 2, 3)
                beta_est = (x - x_best)
                # logger.debug(mean_axis)
                # logger.debug(beta_est)
                # beta_est = beta_est.mean(axis=mean_axis)
                beta_est = np.median(beta_est, axis=mean_axis)
                # logger.debug(beta_est)
            logger.debug(beta.shape, beta_est.shape)
            assert(beta.shape == beta_est.shape)

            return beta_est

        logger.debug("Threshold type: {}, input type: {}"
                     .format(type(threshold), type(inpt)))
        best_th = tf.compat.v1.py_func(mse_optimization, [inpt, threshold], tf.float32)
        best_th.set_shape(threshold.get_shape())

        # TODO in stepwise graph assign might not work because we don't have
        #   a tf variable
        # logger.debug(type(threshold))
        if isinstance(threshold, tf.Variable):
            logger.debug("Yes, Theshold Variable")
            threshold = tf.compat.v1.assign(threshold, best_th)
        else:
            threshold = best_th

        if beta is not None:
            # beta_est = tf.compat.v1.py_func(beta_optimization,
            #                          [inpt, best_th], tf.float32)
            mean_axis = (0, 2, 3) if threshold.shape in [(), (1,)]\
                else (1, 2, 3)
            beta_est = tf.reduce_mean(inpt - alt_inpt, axis=mean_axis)
            # beta_est.set_shape(beta.get_shape())

            if isinstance(beta, tf.Variable):
                logger.debug("Yes, Beta Variable")
                beta = tf.compat.v1.assign(beta, beta_est)
            else:
                beta = beta_est

        threshold = tf.reshape(threshold, threshold_shape)

        quant_factor = half_range / threshold
        val = tf.maximum(tf.minimum(inpts[0], threshold), -threshold)
        val = tf.multiply(val, quant_factor)

        if do_rounding:
            val = tf.round(val)

        # res = tf.cast(val, RtLayerTF.dtype_to_tf[self.dtype])
        res = tf.floor(val)

        # Unquantize
        un_quant_factor = threshold / half_range
        res = res * un_quant_factor

        res_lst = [res]
        if beta is not None:
            res_lst.append(beta)

        return res_lst

    def forward_exec(self, inputs: List[np.ndarray]):
        assert(len(inputs) == len(self.input_shapes))

        feed_dict = {
            self.inpt: inputs[0],
            self.th_params: inputs[1]
        }
        if len(self.input_shapes) > 2:
            feed_dict[self.beta] = inputs[2]
        with tf.Session() as sess:
            return sess.run(self.res, feed_dict=feed_dict)


# MSEQuantizeBias


class MSEQuantizeBiasLayerBase(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic ... TODO

    Arguments
    ---------
    name: str
        the name of this layer
    shape: List[int]/Tuple[int]
        the shape of this layer
    dtype: str
        the type of this layer
    inputs: List[str]
        the input names of this layer
    input_shapes: List[List[int]/Tuple[int]]
        the input shapes for all the inputs
    subgraph: str
        the subgraph to which the layer belongs
    bitwidth: int
        the bitwidth to be quantized to
    do_rounding: bool
        whether to round instead of cast floats to ints
    quantize_layer: bool (unused)
        Indicates whether this should be quantized
    """

    def __init__(self,
                 name,
                 shape,
                 dtype,
                 inputs,
                 input_shapes,
                 subgraph,
                 bitwidth,
                 do_rounding=True):
        super(MSEQuantizeBiasLayerBase, self).__init__(name, 'MSEQuantizeBias',
                                                       shape, dtype, inputs,
                                                       input_shapes, subgraph)

        assert(len(self.input_shapes) == 3)

        if self.dtype not in ['float32']:
            raise ValueError("Invalid threshold training quantize bias layer "
                             " (output) dtype: {}, should be `float32`"
                             .format(self.dtype))

        self.bitwidth = bitwidth
        self.do_rounding = do_rounding

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8"
                                      " for now")

        # INITIALIZE
        self.init()


class MSEQuantizeBiasLayer(MSEQuantizeBiasLayerBase, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        See QuantizeBiasLayer
        """
        logger.info("Init MSE quantize bias layer")

        self.inpt = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.dtype],
            shape=self.input_shapes[0])
        self.th_in = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.dtype],
            shape=self.input_shapes[1])
        self.th_params = tf.compat.v1.placeholder(
            RtLayerTF.dtype_to_tf[self.dtype],
            shape=self.input_shapes[2])
        # logger.info("Threshold_bias shape: {}".format(threshold_bias.shape))

        self.res = self.get_output_tensors([self.inpt, self.th_in,
                                            self.th_params])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert(len(inpts) == 3)
        inpt, threshold_ext, threshold_params = inpts
        bitwidth, do_rounding = self.bitwidth, self.do_rounding

        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1

        sf_ext = tf.divide(threshold_ext, half_range)
        sf_params = tf.divide(tf.abs(threshold_params), half_range)
        sf_acc = tf.multiply(sf_ext, sf_params)
        # logger.debug("sf_acc: {}".format(sf_acc))

        # TODO
        if sf_acc.shape[0] != inpts[0].shape[0]:
            assert(inpts[0].shape[0] % sf_acc.shape[0] == 0)
            logger.warn("[WARNING] Adjusting threshold shape from: {} to {}"
                        " for quantize bias layer."
                        .format(sf_acc.shape[0], inpts[0].shape[0]))
            sf_acc = tf.tile(sf_acc, inpts[0].shape[0] // sf_acc.shape[0])

        # 8 bit only for now
        if bitwidth == 8:
            macc_range = np.power(2, 24)
            macc_half_range = (macc_range / 2) - 1
            th_acc = tf.multiply(sf_acc, macc_half_range)
        # logger.debug("th_acc: {}".format(th_acc))
        logger.debug("Input shape: {}, th_acc shape: {}"
                     .format(inpt.shape, th_acc.shape))
        val = tf.maximum(tf.minimum(inpt, th_acc), -th_acc)
        val = tf.divide(val, sf_acc)

        if do_rounding:
            val = tf.round(val)
        # res = tf.cast(val, RtLayerTF.dtype_to_tf[self.dtype])
        res = tf.floor(val)

        # Unquantize
        un_quant_factor = sf_acc
        res *= un_quant_factor

        return [res]

    def forward_exec(self, inputs: np.ndarray):
        assert(len(inputs) == 3)

        with tf.Session() as sess:
            return sess.run(self.res, feed_dict={
                self.inpt: inputs[0],
                self.th_in: inputs[1],
                self.th_params: inputs[2]})


# MSEMockQuantizeLayer

class MSEMockQuantizeLayerBase(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic ... TODO

    Arguments
    ---------
    name: str
        the name of this layer
    shape: List[int]/Tuple[int]
        the shape of this layer
    dtype: str
        the type of this layer
    inputs: List[str]
        the input names of this layer
    input_shapes: List[List[int]/Tuple[int]]
        the input shapes for all the inputs
    subgraph: str
        the subgraph to which the layer belongs
    axis: int
        specifies the axis on which the scaling should be applied
    bitwidth: int
        the bitwidth to be quantized to
    quantize_layer: bool (unused)
        Indicates whether this should be quantized
    """

    def __init__(self,
                 name,
                 shape,
                 dtype,
                 inputs,
                 input_shapes,
                 subgraph,
                 axis,
                 bitwidth):
        super(MSEMockQuantizeLayerBase, self).__init__(name, 'MSEMockQuantize',
                                                       shape, dtype, inputs,
                                                       input_shapes, subgraph)

        assert(len(self.input_shapes) == 2)
        assert(axis < len(self.input_shapes[0]))

        if not self.dtype == 'float32':
            raise ValueError("The data type of threshold training quantize"
                             " inter layer should be `float32`, but {} was "
                             " provided"
                             .format(self.dtype))

        self.axis = axis
        self.bitwidth = bitwidth

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8"
                                      " for now")

        # INITIALIZE
        self.init()


class MSEMockQuantizeLayer(MSEMockQuantizeLayerBase, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        TODO
        """
        logger.info("Init MSE quantize bias layer")

        self.inpt = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.th = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])
        # logger.info("Threshold_bias shape: {}".format(threshold_bias.shape))

        self.res = self.get_output_tensors([self.inpt, self.th])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert(len(inpts) == 2)
        inpt, threshold_ext = inpts
        bitwidth = self.bitwidth

        return [inpt]

    def forward_exec(self, inputs: List[np.ndarray]):
        assert(len(inputs) == 2)

        with tf.Session() as sess:
            return sess.run(self.res, feed_dict={
                self.inpt: inputs[0],
                self.th: inputs[1]
                })


# MSEQuantizeEltwise


class MSEQuantizeEltwiseLayerBase(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic ... TODO

    Arguments
    ---------
    name: str
        the name of this layer
    shape: List[int]/Tuple[int]
        the shape of this layer
    dtype: str
        the type of this layer
    inputs: List[str]
        the input names of this layer
    input_shapes: List[List[int]/Tuple[int]]
        the input shapes for all the inputs
    subgraph: str
        the subgraph to which the layer belongs
    bitwidth: int
        the bitwidth to be quantized to
    do_rounding: bool
        whether to round instead of cast floats to ints
    mse_opt_num: TODO
    quantize_layer: bool (unused)
        Indicates whether this should be quantized
    """

    def __init__(self,
                 name,
                 shape,
                 dtype,
                 inputs,
                 input_shapes,
                 subgraph,
                 axis,
                 bitwidth,
                 relu,
                 do_rounding=True,
                 mse_opt_num=50):
        super(MSEQuantizeEltwiseLayerBase, self).__init__(
            name, 'MSEQuantizeEltwise', shape, dtype,
            inputs, input_shapes, subgraph)

        assert(len(self.input_shapes) == 5)
        # assert(axis < len(self.input_shapes[0]))

        if not self.dtype == 'float32':
            raise ValueError("The data type of MSEQuantizeMultiple"
                             " layer should be `float32`, but {} was provided"
                             .format(self.dtype))

        self.axis = axis
        self.bitwidth = bitwidth
        self.relu = relu
        self.do_rounding = do_rounding
        self.mse_opt_num = mse_opt_num

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8"
                                      " for now")

        # INITIALIZE
        self.init()


class MSEQuantizeEltwiseLayer(MSEQuantizeEltwiseLayerBase, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        TODO
        """
        logger.info("Init MSEQuantizeEltwise layer")

        self.left = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.left_q = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])
        self.right = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[2])
        self.right_q = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[3])
        self.th = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[4])

        # logger.info("Threshold_bias shape: {}".format(threshold_bias.shape))

        self.res = self.get_output_tensors([self.left, self.left_q, self.right,
                                            self.right_q, self.th])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert(len(inpts) == 5)

        left, left_q, right, right_q, threshold = inpts
        bitwidth, use_relu, do_rounding = \
            self.bitwidth, self.relu, self.do_rounding
        mse_opt_num = self.mse_opt_num

        threshold_shape = [1]*len(self.input_shapes[0])
        # threshold_shape[axis] = int(threshold.shape[0])
        assert int(threshold.shape[0]) == 1
        logger.debug("Threshold_shape: {}".format(threshold_shape))

        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1

        def quantize_unquantize(x, th):
            # Quantize
            th = np.reshape(np.array(th), threshold_shape)
            quant_factor = half_range / th

            qx = np.maximum(np.minimum(x, th), -th)
            qx = qx * quant_factor

            if do_rounding:
                qx = np.round(qx)

            # to int
            qx = np.floor(qx)

            # Unquantize
            un_quant_factor = th / half_range
            res = qx * un_quant_factor

            return res

        def get_MSE(x1_n, x2_n, th_n):
            # nonlocal use_relu
            return MSE(quantize_unquantize(x1_n, th_n), x1_n) + \
                MSE(quantize_unquantize(x2_n, th_n), x2_n)

        def mse_optimization(x1, x1_q, x2, x2_q, th):
            # logger.debug(x.shape, th.shape)
            # TODO #np.isscalar(th) or
            nonlocal mse_opt_num

            logger.debug("mse optimization")
            # axis = None if th.shape in [(), (1,)] else (1,2,3)
            # x1_max = np.max(np.absolute(x1))
            # x2_max = np.max(np.absolute(x2))
            x1_max = np.max(np.absolute(x1_q))
            x2_max = np.max(np.absolute(x2_q))
            # th_max = np.maximum(x1_max, x2_max)
            # th_min = np.minimum(x1_max, x2_max)
            th_max = np.maximum(x1_max, x2_max)

            th_max = np.expand_dims(th_max, axis=0) \
                if th_max.shape == () else th_max
            # logger.debug(th.shape, th_max.shape)
            assert th_max.shape == th.shape

            # mse_th = get_MSE(x1, x2, th)
            # mse_th_max = get_MSE(x1, x2, th_max)
            mse_th = get_MSE(x1_q, x2_q, th)
            mse_th_max = get_MSE(x1_q, x2_q, th_max)

            # logger.debug(type(th), type(mse_th))
            # logger.debug(type(th_max), type(mse_th_max))
            if mse_th <= mse_th_max:
                best_th = th
                best_mse = mse_th
            else:
                best_th = th_max
                best_mse = mse_th_max

            # for _ in range(0, mse_opt_num):
            #     ra = np.random.rand() # np.random.rand(*th.shape)
            #     while ra == 0.:
            #         ra = np.random.rand()
            #     #ra[ra == 0.] = ra.mean() + 0.0001 # replace zeros
            #     th_rand = th_min + ra * (th_max - th_min)

            #     mse_rand = get_MSE(x1, x2, th_rand)
            #     if mse_rand < best_mse:
            #         best_th, best_mse = th_rand, mse_rand

            logger.debug("Name: {}, th: ({}, {}), th_max: ({}, {}), best_th"
                         " ({}, {})".format(self.name, th, mse_th, th_max,
                                            mse_th_max, best_th, best_mse))
            return best_th.astype(np.float32)

        logger.debug("Type threshold: {}, type left: {}, type right: {}"
                     .format(type(threshold), type(left), type(right)))
        best_th = tf.compat.v1.py_func(mse_optimization,
                                [left, left_q, right, right_q, threshold],
                                tf.float32)
        best_th.set_shape(threshold.get_shape())

        # TODO in stepwise graph assign might not work because we don't have
        #   a tf variable
        # logger.debug(type(threshold))
        if isinstance(threshold, tf.Variable):
            logger.debug("Yes, Theshold Variable")
            threshold = tf.compat.v1.assign(threshold, best_th)
        else:
            threshold = best_th

        # Quantize / UnQuantize with new threshold
        quant_factor = half_range / threshold
        val_left = tf.maximum(tf.minimum(left, threshold), -threshold)
        val_left = tf.multiply(val_left, quant_factor)
        val_right = tf.maximum(tf.minimum(right, threshold), -threshold)
        val_right = tf.multiply(val_right, quant_factor)

        if do_rounding:
            val_left = tf.round(val_left)
            val_right = tf.round(val_right)

        val_left = tf.floor(val_left)
        val_right = tf.floor(val_right)

        # Unquantize
        un_quant_factor = threshold / half_range
        res_left = val_left * un_quant_factor
        res_right = val_right * un_quant_factor

        # Eltwise operation
        # shape_for_broadcast = [1, self.shape[1], 1, 1]
        # if res_left.shape[1:] != self.shape[1:]:
        #     res_left = tf.reshape(res_left, shape_for_broadcast)
        # if res_right.shape[1:] != self.shape[1:]:
        #     res_right = tf.reshape(res_right, shape_for_broadcast)

        return [tf.add(res_left, res_right), left_q, right_q, threshold]

    def forward_exec(self, inputs: List[np.ndarray]):
        assert(len(inputs) == 4)

        with tf.Session() as sess:
            return sess.run(self.res, feed_dict={
                self.left: inputs[0],
                self.left_q: inputs[1],
                self.right: inputs[2],
                self.right_q: inputs[3],
                self.th: inputs[4]
                })


# MSEQuantizeConcat
class MSEQuantizeConcatLayerBase(RtLayer):

    __metaclass__ = abc.ABCMeta

    """
    Generic ... TODO

    Arguments
    ---------
    name: str
        the name of this layer
    shape: List[int]/Tuple[int]
        the shape of this layer
    dtype: str
        the type of this layer
    inputs: List[str]
        the input names of this layer
    input_shapes: List[List[int]/Tuple[int]]
        the input shapes for all the inputs
    subgraph: str
        the subgraph to which the layer belongs
    bitwidth: int
        the bitwidth to be quantized to
    do_rounding: bool
        whether to round instead of cast floats to ints
    mse_opt_num: TODO
    quantize_layer: bool (unused)
        Indicates whether this should be quantized
    """

    def __init__(self,
                 name,
                 shape,
                 dtype,
                 layout,
                 inputs,
                 input_shapes,
                 subgraph,
                 axis,
                 bitwidth,
                 do_rounding=True,
                 mse_opt_num=50
                 ):
        super(MSEQuantizeConcatLayerBase, self).__init__(
            name, 'MSEQuantizeConcat', shape, dtype, inputs,
            input_shapes, subgraph)

        assert(len(self.input_shapes) == 3)
        # assert(axis < len(self.input_shapes[0]))

        if not self.dtype == 'float32':
            raise ValueError("The data type of MSEQuantizeMultiple"
                             " layer should be `float32`, but {} was provided"
                             .format(self.dtype))

        self.axis = axis
        self.bitwidth = bitwidth
        self.do_rounding = do_rounding
        self.mse_opt_num = mse_opt_num

        if self.bitwidth != 8:
            raise NotImplementedError("Quantize layer only supports bitwith 8"
                                      " for now")

        # INITIALIZE
        self.init()


class MSEQuantizeConcatLayer(MSEQuantizeConcatLayerBase, RtLayerTF):

    def init(self):
        # type: () -> None
        """
        TODO
        """
        logger.info("Init MSEQuantizeEltwise layer")

        self.left = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[0])
        self.right = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[1])
        self.th = \
            tf.compat.v1.placeholder(RtLayerTF.dtype_to_tf[self.dtype],
                                     shape=self.input_shapes[2])

        # logger.info("Threshold_bias shape: {}".format(threshold_bias.shape))

        self.res = self.get_output_tensors([self.left, self.right, self.th])[0]
        logger.info("Res shape: {}".format(self.res.shape))

    def get_output_tensors(self, inpts: List[tf.Tensor], **kwargs) -> tf.Tensor:
        assert(len(inpts) == 3)

        left, right, threshold = inpts
        axis, bitwidth, do_rounding = \
            self.axis, self.bitwidth, self.do_rounding
        mse_opt_num = self.mse_opt_num

        threshold_shape = [1]*len(self.input_shapes[0])
        threshold_shape[axis] = int(threshold.shape[0])
        logger.debug("Threshold_shape: {}".format(threshold_shape))

        max_range = np.power(2, bitwidth)
        half_range = (max_range / 2.) - 1

        def quantize_unquantize(x, th):
            # Quantize
            th = np.reshape(np.array(th), threshold_shape)
            quant_factor = half_range / th

            qx = np.maximum(np.minimum(x, th), -th)
            qx = qx * quant_factor

            if do_rounding:
                qx = np.round(qx)

            qx = np.floor(qx)  # to int

            # Unquantize
            un_quant_factor = th / half_range
            res = qx * un_quant_factor

            return res

        def get_MSE(x1_n, x2_n, th_n):
            return MSE(quantize_unquantize(x1_n, th_n), x1_n) + \
                MSE(quantize_unquantize(x2_n, th_n), x2_n)

        def mse_optimization(x1, x2, th):
            # logger.debug(x.shape, th.shape)
            # TODO #np.isscalar(th) or
            nonlocal mse_opt_num

            logger.debug("mse optimization")
            # axis = None if th.shape in [(), (1,)] else (1,2,3)
            x1_max = np.max(np.absolute(x1))
            x2_max = np.max(np.absolute(x2))
            th_max = np.maximum(x1_max, x2_max)
            th_max = np.expand_dims(th_max, axis=0) if th_max.shape == ()\
                else th_max
            # logger.debug(th.shape, th_max.shape)
            assert(th_max.shape == th.shape)  # or \
            # (th_max.shape in [(), (1,)] and th.shape in [(), (1,)]))

            mse_th = get_MSE(x1, x2, th)
            mse_th_max = get_MSE(x1, x2, th_max)
            logger.debug(self.name, mse_th, th, mse_th_max, th_max)
            # logger.debug(type(th), type(mse_th))
            # logger.debug(type(th_max), type(mse_th_max))
            if mse_th <= mse_th_max:
                best_th = th
                best_mse = mse_th
            else:
                best_th = th_max
                best_mse = mse_th_max

            # for _ in range(0, mse_opt_num):
            #     ra = np.random.rand() # np.random.rand(*th.shape)
            #     while ra == 0.:
            #         ra = np.random.rand()
            #     #ra[ra == 0.] = ra.mean() + 0.0001 # replace zeros
            #     th_rand = ra * th_max

            #     mse_rand = get_MSE(x1, x2, th_rand)
            #     if mse_rand < best_mse:
            #         best_th, best_mse = th_rand, mse_rand

            return best_th.astype(np.float32)

        logger.debug(type(threshold), type(left), type(right))
        best_th = tf.compat.v1.py_func(mse_optimization, [left, right, threshold],
                                tf.float32)
        best_th.set_shape(threshold.get_shape())

        # TODO in stepwise graph assign might not work because we don't have
        #   a tf variable
        # logger.debug(type(threshold))
        if isinstance(threshold, tf.Variable):
            logger.debug("Yes, Theshold Variable")
            threshold = tf.compat.v1.assign(threshold, best_th)
        else:
            threshold = best_th

        # Eltwise operation
        shape_for_broadcast = [1, self.shape[1], 1, 1]
        if left.shape[1:] != self.shape[1:]:
            left = tf.reshape(left, shape_for_broadcast)
        if right.shape[1:] != self.shape[1:]:
            right = tf.reshape(right, shape_for_broadcast)

        return [tf.add(left, right), threshold]

    def forward_exec(self, inputs: List[np.ndarray]):
        assert(len(inputs) == 4)

        with tf.Session() as sess:
            return sess.run(self.res, feed_dict={
                self.left: inputs[0],
                self.right: inputs[1],
                self.th: inputs[2]
                })

# Quantize layer building functions


# TODO make better registration
def get_mse_quantize_layer(X, input_shapes, params, **kwargs):
    # (XLayer, dict, dict, QuantParams)
    #   -> List[rt_layer.RtLayer]
    """
    TODO formalize checks
    """
    bitwidth = X.attrs['quant_bitwidth']
    axis = X.attrs['axis']
    dtype = X.attrs['dtype']
    mse_opt_num = X.attrs['mse_opt_num']

    assert(len(X.bottoms) in [2, 4])
    assert(bitwidth == 8)
    assert(dtype == 'float32')
    assert(axis in [0, 1, 2, 3])

    return [MSEQuantizeLayer(
        name=X.name,
        shape=X.shapes[:],
        dtype=dtype,
        inputs=X.bottoms,
        input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
        subgraph=X.subgraph,
        axis=axis,  # TODO
        bitwidth=bitwidth,
        do_rounding=True,
        mse_opt_num=mse_opt_num
    )]


def get_mse_quantize_bias_layer(X, input_shapes, params, **kwargs):
    # (XLayer, dict, dict, QuantParams)
    #   -> List[rt_layer.RtLayer]
    """
    TODO formalize checks
    """

    bitwidth = X.attrs['quant_bitwidth']
    dtype = X.attrs['dtype']

    assert(len(X.bottoms) == 3)
    assert(bitwidth == 8)
    assert(dtype == 'float32')

    return [MSEQuantizeBiasLayer(
        name=X.name,
        shape=X.shapes[:],
        dtype=dtype,  # 'int32',
        inputs=X.bottoms,
        input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
        subgraph=X.subgraph,
        bitwidth=bitwidth,
        do_rounding=True
    )]


def get_mse_mock_quantize_layer(X, input_shapes, params, **kwargs):
    # (XLayer, dict, dict, QuantParams)
    #   -> List[rt_layer.RtLayer]
    """
    TODO formalize checks
    """
    bitwidth = X.attrs['quant_bitwidth']
    axis = X.attrs['axis']
    dtype = X.attrs['dtype']

    assert(len(X.bottoms) == 2)
    assert(bitwidth == 8)
    assert(dtype == 'float32')
    assert(axis in [0, 1, 2, 3])

    return [MSEMockQuantizeLayer(
        name=X.name,
        shape=X.shapes,
        dtype=dtype,
        inputs=X.bottoms,
        input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
        subgraph=X.subgraph,
        axis=axis,  # TODO
        bitwidth=bitwidth  # TODO
    )]

# def get_mse_eltwise_scale_quantize_layer(X, input_shapes, params, **kwargs):
#     # (XLayer, dict, dict, QuantParams)
#     #   -> List[rt_layer.RtLayer]
#     """
#     TODO formalize checks
#     """
#     bitwidth = X.attrs['quant_bitwidth']
#     #axis = X.attrs['axis']
#     dtype = X.attrs['dtype']
#     #mse_opt_num = X.attrs['mse_opt_num']

#     assert(len(X.bottoms) in [4])
#     assert(bitwidth == 8)
#     assert(dtype == 'float32')
#     #assert(axis in [0,1,2,3])

#     return [MSEEltwiseScaleQuantizeLayer(
#         name=X.name,
#         shape=X.shapes,
#         dtype=dtype,
#         inputs=X.bottoms,
#         input_shapes=TupleShape([input_shapes[bottom]
#                                  for bottom in X.bottoms],
#         subgraph=X.subgraph,
#         bitwidth=bitwidth,
#         do_rounding=True
#     )]


def get_mse_quantize_eltwise_layer(X, input_shapes, params, **kwargs):
    # (XLayer, dict, dict, QuantParams) -> List[rt_layer.RtLayer]
    """
    TODO formalize checks
    """
    bitwidth = X.attrs['quant_bitwidth']
    axis = X.attrs['axis']
    dtype = X.attrs['dtype']
    mse_opt_num = X.attrs['mse_opt_num']

    use_relu = 'activation' in X.attrs and X.attrs['activation'] == 'ReLU'

    assert len(X.bottoms) in [5]
    assert bitwidth == 8
    assert dtype == 'float32'
    assert axis in [0, 1, 2, 3]

    layers = [MSEQuantizeEltwiseLayer(
        name=X.name,
        shape=X.shapes[:],
        dtype=dtype,
        inputs=X.bottoms,
        input_shapes=[input_shapes[bottom] for bottom in X.bottoms],
        subgraph=X.subgraph,
        axis=axis,  # TODO
        bitwidth=bitwidth,
        relu=use_relu,
        do_rounding=True,
        mse_opt_num=mse_opt_num
    )]

    if use_relu:
        layers.append(ReluLayer(
            name=X.name,
            xtype='ReLU',
            shape=X.shapes,
            dtype='float32',
            inputs=[X.name],
            input_shapes=TupleShape([X.shapes[:]]),
            subgraph=X.subgraph,
            attrs={}
        ))

    return layers


X_2_TF.update({
    'MSEQuantize': get_mse_quantize_layer,
    'MSEQuantizeBias': get_mse_quantize_bias_layer,
    'MSEMockQuantize': get_mse_mock_quantize_layer,
    # 'MSEEltwiseScaleQuantize': get_mse_eltwise_scale_quantize_layer,
    'MSEQuantizeEltwise': get_mse_quantize_eltwise_layer
})
