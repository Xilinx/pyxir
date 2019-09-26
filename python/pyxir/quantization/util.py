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
Module for quantization utility functions
"""

import numpy as np

import logging
logger = logging.getLogger("pyxir")


def Float2Fixed2Float(data, bitwidth, threshold, f):
    if np.isclose(threshold, 0.):
        threshold = np.zeros_like(threshold)
    scaling_factor = threshold / (np.power(2.0, bitwidth - 1) - 1)
    orig = np.array(data)
    data = np.clip(data, -threshold, threshold)
    if threshold != 0:
        data /= scaling_factor
        data = f(data)
        data *= scaling_factor
    error = np.sum(np.square(orig-data))
    # logger.debug("Square Error: {}".format(error))
    return data


def CdfMeasure(x, y, measure_name):
    if False:
        pass
    elif measure_name == "Kullback-Leibler-J":
        return np.sum((x - y) * np.log2(x / y))
    else:
        return CdfMeasure(x, y, "Kullback-Leibler-J")


def ComputeThreshold(data, bitwidth, bins):
    mn = 0
    mx = np.abs(data).max()
    logger.debug("Min: {}, max: {}".format(mn, mx))
    zed = np.float64(0.0)
    if np.isclose(mx, zed):
        th_layer_out = zed
        sf_layer_out = zed
        logger.debug("Mean : th_layer_out: {}, sf_layer_out: {}"
            .format(th_layer_out, sf_layer_out))
        return th_layer_out
    hist, bin_edges = np.histogram(np.abs(data), bins, range = (mn, mx), density = True)
    hist = hist / np.sum(hist)
    cumsum = np.cumsum(hist)

    n = pow(2, bitwidth - 1)
    threshold = []
    scaling_factor = []
    d = []

    logger.debug("n: {}, len(bin_edges): {}".format(n, len(bin_edges)))

    if n + 1 > len(bin_edges) - 1:
        th_layer_out = bin_edges[-1]
        sf_layer_out = th_layer_out / (np.power(2.0, bitwidth - 1) - 1)
        logger.debug("Mean : th_layer_out: {}, sf_layer_out: {}"
            .format(th_layer_out, sf_layer_out))
        return th_layer_out

    zero_in_q = False
    for i in range(n + 1, len(bin_edges), 1):
        threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
        threshold = np.concatenate((threshold, [threshold_tmp]))
        scaling_factor_tmp = threshold_tmp / (np.power(2.0, bitwidth - 1) - 1)
        scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
        
        p = np.copy(cumsum)
        p[i-1:] = 1
        
        x = np.linspace(0., 1., n)
        xp = np.linspace(0., 1., i)
        fp = p[:i]
        p_interp = np.interp(x, xp, fp)

        x = np.linspace(0., 1., i)
        xp = np.linspace(0., 1., n)
        fp = p_interp
        q_interp = np.interp(x, xp, fp)

        q = np.copy(p)
        q[:i] = q_interp

        # Check if 0. is in q, if so and we try to calculate the
        #   KL divergence, this will give a division by zero. This
        #   happens for the input layer of some networks or with a
        #   batch norm after the input layer. Therefore, if this
        #   happens, we will use the maximum value of the input
        #   as threshold.
        if 0. in q:
            zero_in_q = True
            break
        d_tmp = CdfMeasure(cumsum, q, "Kullback-Leibler-J")
        d = np.concatenate((d, [d_tmp]))
    if not zero_in_q:
        logger.debug("Zero in KL q")
        th_layer_out = threshold[np.argmin(d)]
        sf_layer_out = scaling_factor[np.argmin(d)]
    else:
        th_layer_out = ThresholdLayerInputs(data, bitwidth).astype(np.float64)
        sf_layer_out = th_layer_out / (np.power(2.0, bitwidth - 1) - 1)

    logger.debug("Mean : th_layer_out: {}, sf_layer_out: {}"
                 .format(th_layer_out, sf_layer_out))

    assert type(th_layer_out) == np.float64
    return th_layer_out


# uses min/max thresholds
def ThresholdLayerInputs(data, bitwidth):
    threshold = np.max(np.abs(data))
    #   threshold = np.power(2, np.ceil(np.log2(threshold / (np.power(2, bitwidth - 1) - 1)))) * (np.power(2, bitwidth - 1) - 1)
    return threshold


# uses min/max thresholds assume first dim is channels...
def ThresholdWeights(data, bitwidth):
    # logger.debug("Weights array shape".format(data.shape))
    #    threshold = np.repeat(np.max(np.abs(data), axis=tuple(range(0, data.ndim))), data.shape[0])
    threshold = np.max(np.abs(data), axis=tuple(range(1, data.ndim)))
    #    threshold = np.where(np.isclose(threshold, np.zeros_like(threshold)), np.zeros_like(threshold), threshold)
    #    threshold = np.power(2, np.ceil(np.log2(threshold / (np.power(2, bitwidth - 1) - 1)))) * (np.power(2, bitwidth - 1) - 1)
    return threshold


# uses min/max thresholds assume first dim is channels...
def ThresholdBiases(data, bitwidth):
    #    threshold = np.repeat(np.max(np.abs(data), axis=tuple(range(0, data.ndim))), data.shape[0])
    threshold = np.max(np.abs(data), axis=tuple(range(1, data.ndim)))
    #    threshold = np.where(np.isclose(threshold, np.zeros_like(threshold)), np.zeros_like(threshold), threshold)
    #    threshold = np.power(2, np.ceil(np.log2(threshold / (np.power(2, bitwidth - 1) - 1)))) * (np.power(2, bitwidth - 1) - 1)
    return threshold


# uses entropy for threshold... returns scalar
def ThresholdLayerOutputs(data, bitwidth):
    threshold = ComputeThreshold(data.flatten(), bitwidth, "sqrt")
    #    threshold = np.power(2, np.ceil(np.log2(threshold / (np.power(2, bitwidth - 1) - 1)))) * (np.power(2, bitwidth - 1) - 1)
    return threshold


# single threshold for entire blob
# numpy in, numpy out - all floating point
def QuantizeBlob(data, bitwidth):
    threshold = ThresholdLayerOutputs(data, bitwidth)
    return (Float2Fixed2Float(data, bitwidth, threshold, np.round), threshold)


# given a previously computed threshold scalar, recompute for this blob
def QuantizeThresholdBlob(data, bitwidth, threshold):
    # threshold should be a scalar...
    assert type(threshold) in [np.float32, np.float64], \
        "Theshold is not a scalar"
    return Float2Fixed2Float(data, bitwidth, threshold, np.round)    


# given a previous computed threshold vector, recompute for this blob
# NOTE: This only works for CAFFE weight blobs currently...
# CAFFE Format = (outch,inch,kheight,kwidth)
# TF Format = (kheight,kwidth,inch,outch)
def QuantizeWeights(threshold, bitwidth, data, mode='caffe'):
    if mode == 'tf':
        data = data.transpose(2, 3, 1, 0)

    assert data.shape[0] == threshold.shape[0], \
        "Threshold shape does not match weight data shape"

    for i in range(len(threshold)):
        # logger.debug("Quantizing conv weight channel {} ...".format(i))
        data[i] = Float2Fixed2Float(data[i], bitwidth, threshold[i], np.round)    

    if mode == 'tf':
        data = data.transpose(3, 2, 0, 1)

    return data
