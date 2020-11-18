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
Module for transforming Relay L4 operators to XLayer objects

L4: Broadcast and reduction operations
"""

import math
import logging
import numpy as np
import pyxir as px

import tvm

from pyxir import graph
from pyxir.graph.layer import xlayer_factory as xlf

from .util import broadcast_shapes
from .relay_2_xlayer_registry import register_relay_2_xlayer_converter,\
    register_relay_2_xlayer_converter_base

logger = logging.getLogger("pyxir")


@register_relay_2_xlayer_converter_base('greater')
def greater(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    Compare two input layers

    Relay
    -----
    Type: tvm.relay.greater
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - lhs (relay.Expr)
            The left hand side input data
        - rhs (relay.Expr)
            The right hand side input data
    """

    X = px.ops.greater(op_name, in_xlayers, relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter('mean')
def mean(expr, params, schedule, net, op_idx, RELAY_2_XLAYER, **kwargs):
    # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
    #   Dict[int, XLayer], Dict[str, int], Dict[str, Function]) -> XLayer
    """
    TODO

    Relay
    -----
    Type: tvm.relay.op.reduce.mean
    Ref: https://docs.tvm.ai/api/python/relay/nn.html
    Parameters:
        - data (tvm.relay.Expr)
            The input data to the operator.
        - axis (None or int or tuple of int)
            Axis or axes along which a mean operation is performed. The
            default, axis=None, will compute the mean of all elements in
            the input array. If axis is negative it counts from the last
            to the first axis.
        - keepdims (bool)
            If this is set to True, the axes which are reduced are left in
            the result as dimensions with size one. With this option, the
            result will broadcast correctly against the input array.
        - exclude (bool)
            If exclude is true, reduction will be performed on the axes that
            are NOT in axis instead.
    """
    if expr in net:
        logger.debug("MEMORY: MEAN")
        # This expressions is already transformed so we reuse that one
        return net[expr]

    expr_axis = expr.attrs.axis
    if expr_axis is None:
        axis = None
    elif isinstance(expr_axis, int):
        axis = [expr_axis]
    else:
        axis = [int(e) for e in list(expr_axis)]
    keepdims = bool(expr.attrs.keepdims)
    exclude = bool(expr.attrs.exclude)

    data_expr, data_expr_class = expr.args[0], expr.args[0].__class__.__name__

    data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params, schedule,
                                                 net, op_idx, RELAY_2_XLAYER,
                                                 **kwargs)

    logger.debug("mean: {}".format(""))

    # Update schedule with input data layer
    if data_expr not in net:
        schedule.append(data_expr)
        net[data_expr] = data_layer

    # Create XLayer
    op_name = 'mean-' + str(hash(expr))

    X = xlf.get_xop_factory_func('Mean')(op_name, data_layer,
                                         axis, keepdims, exclude,
                                         relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    # !Important: set input layer tops:
    data_layer.tops.append(op_name)

    return X


@register_relay_2_xlayer_converter_base('strided_slice')
def strided_slice(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    Strided slice

    Relay
    -----
    Type: tvm.relay.strided_slice
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - data (relay.Expr)
            The source array to be sliced.
        - begin (relay.Expr, Tuple[int], or List[int])
            The indices to begin with in the slicing.
        - end (relay.Expr, Tuple[int], or List[int])
            Indices indicating end of the slice.
        - strides (relay.Expr, Tuple[int], or List[int], optional)
            Specifies the stride values, it can be negative in that case, the
            input tensor will be reversed in that particular axis.
        - slice_mode (str, optional)
            The slice mode [end, size]. end: The ending indices for the slice [default].
            size: The input strides will be ignored, input end in this mode indicates the
            size of a slice starting at the location specified by begin. If end[i] is -1,
            all remaining elements in that dimension are included in the slice.
    """
    begin = [int(e) for e in list(expr.attrs.begin)]
    end = [int(e) for e in list(expr.attrs.end)]
    expr_strides = list(expr.attrs.strides)
    if expr_strides is None:
        strides = [1] * len(begin)
    elif len(expr_strides) == 1:
        strides = [int(expr_strides[0]) for _ in begin]
    else:
        strides = [int(e) for e in list(expr.attrs.strides)]
    slice_mode = expr.attrs.slice_mode if expr.attrs.slice_mode is not None else 'end'

    X = px.ops.strided_slice(op_name, in_xlayers, begin=begin, end=end, strides=strides,
                             slice_mode=slice_mode, relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter_base('where')
def where(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    Selecting elements from input layers input layers

    Relay
    -----
    Type: tvm.relay.where
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - condition (relay.Expr)
            Where True, yield x, otherwise yield y
        - x (relay.Expr)
            The first array or scalar to be selected.
        - y (relay.Expr)
            The second array or scalar to be selected.
    """
    lshape = list(in_xlayers[0].shapes[:])
    rshape = list(in_xlayers[1].shapes[:])
    shape = broadcast_shapes(lshape, rshape)

    X = px.ops.any_op(op_name, in_xlayers, any_shape=shape, relay_id=[hash(expr)])

    return X