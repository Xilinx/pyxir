
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
Module for transforming Relay L3 operators to XLayer objects

L3: Additional math and transform operators
"""

import math
import logging
import numpy as np
import pyxir as px

from math import ceil
from typing import Dict, List, Callable

import tvm
from tvm.relay.expr import Expr

from pyxir import graph
from pyxir.graph.layer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf

from .util import Schedule
from .relay_2_xlayer_registry import register_relay_2_xlayer_converter,\
    register_relay_2_xlayer_converter_base

logger = logging.getLogger("pyxir")



@register_relay_2_xlayer_converter_base('tile')
def tile(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    TVM tile to XLayer

    Relay
    -----
    Type: tvm.relay.tile
    Ref: https://tvm.apache.org/docs/api/python/relay/index.html#tvm.relay.tile
    Parameters:
        - reps
            The number of times repeating the tensor
    """

    assert len(in_xlayers) == 1
    
    reps        = [int(r) for r in expr.attrs['reps']]
    input_shape = [int(s) for s in expr.type_args[0].shape]
    newshape    = [ r*s for r,s in zip(reps,input_shape) ]

    logger.debug("tile: {}".format(op_name))
    X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))
    
    return X

@register_relay_2_xlayer_converter_base('full')
def full(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    TVM full to XLayer

    Relay
    -----
    Type: tvm.relay.full
    Ref: https://tvm.apache.org/docs/api/python/relay/index.html#tvm.relay.full
    Parameters:
        - fill_value
            The value to fill. Must be scalar
        - shape
            The shape of the target
        - dtype (str, optional)
            The target data type.
    """

    assert len(in_xlayers) == 1
    
    newshape   = [ int(dim) for dim in expr.attrs.shape ]

    # currently not used
    dtype      = expr.attrs.dtype
    fill_value = expr.args[0].data.asnumpy() 
    value      = np.full(newshape,fill_value,dtype)

    logger.debug("full: {}".format(op_name))
    X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))
    
    return X

@register_relay_2_xlayer_converter_base('arange')
def arange(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    TVM arange to XLayer

    Relay
    -----
    Type: tvm.relay.arange
    Ref: https://tvm.apache.org/docs/api/python/relay/index.html#tvm.relay.arange
    Parameters:
        - start (tvm.Expr, optional)
            Start of interval. The interval includes this value. The default start value is 0.
        - stop (tvm.Expr)
            Stop of interval. The interval does not include this value.
        - step (tvm.Expr, optional)
            Spacing between values. The default step size is 1.
        - dtype (str, optional)
            The target data type.
    """
    assert len(in_xlayers) in [1, 2, 3]

    if len(in_xlayers) == 1 and 'Constant' in in_xlayers[0].type:
        newshape = [int(in_xlayers[0].data[0])]
    elif len(in_xlayers) == 2\
            and 'Constant' in in_xlayers[0].type\
            and 'Constant' in in_xlayers[1].type:
        begin = int(in_xlayers[0].data[0])
        end = int(in_xlayers[1].data[0])
        newshape = [end - begin]
    elif len(in_xlayers) == 3\
            and 'Constant' in in_xlayers[0].type\
            and 'Constant' in in_xlayers[1].type\
            and 'Constant' in in_xlayers[2].type:
        begin = int(in_xlayers[0].data[0])
        end = int(in_xlayers[1].data[0])
        step = float(in_xlayers[2].data[0])
        newshape = [int(ceil((end - begin) / step))]
    else:
        newshape = [-1]

    X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter_base('cast')
def cast(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Cast the input tensor to the specified data type

    Relay
    -----
    Type: tvm.relay.cast
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - data (relay.Expr)
            The input data to the operator.
        - dtype (str)
            The target data type
    """
    dtype = str(expr.attrs.dtype)
    X = px.ops.cast(op_name, in_xlayers, dtype=dtype, relay_id=[hash(expr)])
    return X


@register_relay_2_xlayer_converter_base('clip')
def cast(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Conversion of Relay 'clip' layer

    Relay
    -----
    Type: tvm.relay.op.clip
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - a (relay.Expr)
            The input tensor.
        - a_min (float)
            The clip minimum.
        - a_max (float)
            The clip maximum.
    """
    a_min = float(expr.attrs.a_min)
    a_max = float(expr.attrs.a_max)
    logger.debug("clip: {}".format(op_name))
    X = px.ops.clip(op_name, in_xlayers[0], a_min, a_max, relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))
    return X


@register_relay_2_xlayer_converter_base('ones_like')
def ones_like(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Ones like to XLayer

    Relay
    -----
    Type: tvm.relay.ones_like
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - data (relay.Expr)
            The input data
    """
    assert len(in_xlayers) == 1
    newshape = list(in_xlayers[0].shapes[:])
    X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])
    return X


@register_relay_2_xlayer_converter_base('nn.leaky_relu')
def nn_leaky_relu(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    TVM leaky rectified linear unit nonlinearity to XLayer

    Relay
    -----
    Type: tvm.relay.nn.leaky_relu
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (tvm.relay.Expr)
            The input data to the operator.
        - alpha (float)
            Slope coefficient for the negative half axis.
    """
    alpha = float(expr.attrs.alpha)
    X = px.ops.leaky_relu(op_name, in_xlayers, alpha=alpha, relay_id=[hash(expr)])
    return X


@register_relay_2_xlayer_converter_base('nn.prelu')
def nn_prelu(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Compute parameterized rectified linear unit nonlinearity

    Relay
    -----
    Type: tvm.relay.nn.prelu
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (tvm.relay.Expr)
            The input data to the operator.
        - alpha (tvm.relay.Expr)
            Slope coefficient for the negative half axis.
        - axis (int, optional)
            Specify which shape axis the channel is specified.
    """
    # assert len(in_xlayers) == 2
    # axis = int(expr.attrs.axis) if expr.attrs.axis is not None else 1
    # X = px.ops.prelu(op_name, in_xlayers[0], alpha, axis, relay_id=[hash(expr)])
    # return X
    raise NotImplementedError("Relay Parametric ReLU to XLayer not implemented")


@register_relay_2_xlayer_converter_base('repeat')
def repeat(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    TVM: Repeats elements of an array. By default, repeat flattens the
    input array into 1-D and then repeats the elements.

    Relay
    -----
    Type: tvm.relay.repeat
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - data (relay.Expr)
            The input
        - repeats (int)
            The number of repetitions for each element.
        - Axis (int)
            The axis along which to repeat values. The negative numbers are
            interpreted counting from the backward. By default, use the flattened
            input array, and return a flat output array.
    """
    repeats = int(expr.attrs.repeats)
    axis = int(expr.attrs.axis) if expr.attrs.axis else None
    in_shape = list(in_xlayers[0].shapes[:])

    if axis is None or axis == 0:
        shape = [int(np.prod(in_shape)) * repeats]
    else:
        shape = in_shape
        shape[axis] = in_shape[axis] * repeats

    X = px.ops.any_op(op_name, in_xlayers, any_shape=shape, relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter('reshape')
def reshape(expr: Expr,
            params: Dict[str, np.ndarray],
            schedule: Schedule,
            net: Dict[Expr, Expr],
            op_idx: Dict[str, int],
            RELAY_2_XLAYER: Dict[str, Callable],
            **kwargs) -> XLayer:
    """
    Relay Reshape to XLayer converter

    Relay
    -----
    Type: tvm.relay.op.transform.reshape
    Ref: https://docs.tvm.ai/api/python/relay/op.html
    Parameters:
        - data (relay.Expr)
            The input data to the operator.
        - newshape (Union[int, Tuple[int], List[int]])
            The new shape. Should be compatible with the original shape.
    """
    if expr in net:
        logger.debug("MEMORY: RESHAPE")
        # This expressions is already transformed so we reuse that one
        return net[expr]

    relayshape = [int(e) for e in list(expr.attrs.newshape)]

    data_expr, data_expr_class = expr.args[0], expr.args[0].__class__.__name__
    data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params, schedule,
                                                 net, op_idx, RELAY_2_XLAYER,
                                                 **kwargs)

    logger.debug("reshape: {}".format(hash(expr)))
    logger.debug("relay shape: {}".format(relayshape))
    # Parse the Relay newshape list because it can contain special numbers
    # (https://docs.tvm.ai/api/python/relay/op.html#tvm.relay.op.transform.reshape)
    # TODO TEST
    input_shape = list(data_layer.shapes)

    newshape = []
    i, j = 0, 0  # i is index in relayshape, j in input_shape
    while i < len(relayshape):
        dim = relayshape[i]
        if dim == 1 and i == 0: # ex: relayshape -> [1, -1] input_shape -> [10,1]
            newshape.append(dim)
            j -=1
        elif dim > 0:
            newshape.append(dim)
        elif dim == 0:
            newshape.append(input_shape[j])
        elif dim == -1 and i == 0 and input_shape[0] == -1:
            newshape.append(-1)
        elif dim == -1 and i == 0 and len(input_shape) == 1: # ex: relayshape -> [-1,1] input_shape -> [10]
            newshape.append(input_shape[0])
        elif dim == -1:
            newshape.append(int(np.prod(input_shape[j:]) / np.prod(relayshape[i+1:])))
        elif dim == -2:
            newshape.extend(input_shape[j:])
        elif dim == -3:
            newshape.append(input_shape[j]*input_shape[j+1])
            j += 1
        elif dim == -4:
            assert(i < (len(relayshape) - 2))
            nxt = relayshape[i+1]
            nxtnxt = relayshape[i+2]
            if nxt == -1 and nxtnxt == -1:
                raise ValueError("Invalid sequence in relay reshape operator"
                                 " newshape attribute: [-4,-1,-1].")
            elif nxt == -1:
                nxt = input_shape[j] / nxtnxt
            elif nxtnxt == -1:
                nxtnxt = input_shape[j] / nxt
            assert(input_shape[j] == nxt * nxtnxt)
            newshape.extend([int(nxt), int(nxtnxt)])
            i += 2
        else:
            raise ValueError("Only integers greater or equal to -4 are allowed"
                             " in Relay reshape operator. [-4,-3,-2,-1,0] are"
                             " special cases. See https://docs.tvm.ai/api/"
                             "python/relay/op.html#tvm.relay.op.transform."
                             "reshape")
        i += 1
        j += 1

    logger.debug("-- newshape: {}".format(newshape))
    if list(data_layer.shapes)[0] == -1:
        assert abs(np.prod(list(data_layer.shapes))) % np.prod(newshape) == 0
        newshape[0] = -1
    else:
        assert np.prod(list(data_layer.shapes)) == np.prod(newshape),\
            "Incompatible shapes for: input shape is: {} and target shape is: {}"\
            .format(list(data_layer.shapes), newshape)

    # Create XLayer
    op_name = 'reshape-' + str(hash(expr))

    X = px.ops.reshape(op_name, data_layer, newshape, relay_id=[hash(expr)])

    # Otherwise precompute
    if X.name != data_layer.name:
        # Update schedule with input data layer
        if data_expr not in net:
            schedule.append(data_expr)
            net[data_expr] = data_layer

        # !Important: set input layer tops:
        data_layer.tops.append(op_name)

    return X


@register_relay_2_xlayer_converter_base('split')
def split(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Split the input tensor along specified axis by the provided indices

    Relay
    -----
    Type: tvm.relay.split
    Desc:
        Split input tensor along axis by sections or indices.

        If indices_or_sections is an integer, the input will be divided
        equally along given axis. If such a split is not possible, an
        error is raised.

        If indices_or_sections is a tuple of sorted integers, the entries
        indicate where along axis the array is split.

    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (relay.Expr)
            The source array.
        - indices_or_sections (int or tuple of int)
            Indices or sections to split into. Accepts an int or a tuple
        - axis (int, optional)
            The axis over which to split.
    """
    i_or_s = expr.attrs.indices_or_sections
    if isinstance(i_or_s, tvm.tir.expr.IntImm):
        indices = int(i_or_s)
    else:
        indices = list([int(e) for e in list(i_or_s)])
    axis = int(expr.attrs.axis) if expr.attrs.axis is not None else 0

    X = px.ops.split(op_name, in_xlayers, axis=axis, indices=indices, relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))
    return X

@register_relay_2_xlayer_converter_base('squeeze')
def squeeze(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Relay squeeze to XLayer converter

    Relay
    -----
    Type: tvm.relay.op.transform.squeeze
    Ref: https://docs.tvm.ai/api/python/relay/nn.html
    Parameters:
        - data (tvm.relay.Expr)
            The input data to the operator.
        - axis (None or List[int])
            The set of axes to remove. If axis = None, remove all
            axis of dimensions 1. If any specified axis has dimension
            that does not equal 1, it is an error.
    """
    expr_axis = expr.attrs.axis
    axis = [int(e) for e in list(expr_axis)] if expr_axis is not None else None
    X = px.ops.squeeze(op_name, in_xlayers[0], axis, relay_id=[hash(expr)])
    return X


@register_relay_2_xlayer_converter_base('take')
def take(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Relay Take to XLayer

    Relay
    -----
    Type: tvm.relay.op.transform.squeeze
    Ref: https://docs.tvm.ai/api/python/relay/nn.html
    Parameters:
        - data (tvm.relay.Expr)
            The input data to the operator.
        - indices (rely.Expr)
            The indices of the values to extract.
        - axis (int, optional)
            The axis over which to select values. By default, the flattened
            input array is used.
        - mode (str, optional)
            Specifies how out-of-bound indices will behave [clip, wrap, fast].
            clip: clip to the range (default). wrap: wrap around the indices.
            fast: no clip or wrap around (user must make sure indices are
            in-bound).
    """
    axis = int(expr.attrs.axis)
    mode = str(expr.attrs.mode)
    X = px.ops.take(op_name, in_xlayers, axis=axis, mode=mode, relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))
    return X

# TODO Move Transpose to this API
# @register_relay_2_xlayer_converter_base('transpose')
# def transpose(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
#     """
#     Relay Transpose to XLayer

#     Relay
#     -----
#     Type: tvm.relay.op.transform.squeeze
#     Ref: https://docs.tvm.ai/api/python/relay/nn.html
#     Parameters:
#         - data (tvm.relay.Expr)
#             The input data to the operator.
#         - indices (rely.Expr)
#             The indices of the values to extract.
#         - axis (int, optional)
#             The axis over which to select values. By default, the flattened
#             input array is used.
#         - mode (str, optional)
#             Specifies how out-of-bound indices will behave [clip, wrap, fast].
#             clip: clip to the range (default). wrap: wrap around the indices.
#             fast: no clip or wrap around (user must make sure indices are
#             in-bound).
#     """
#     assert len(in_xlayers) == 1, "Transpose expects one input layer"
#     expr_axes = expr.attrs.axes
#     axes = [int(e) for e in list(expr_axes)] if expr_axes is not None else None
#     data_layer = in_xlayers[0]

#     X = px.ops.transpose(op_name, in_xlayers, axis=axis, mode=mode, relay_id=[hash(expr)])
#     logger.debug("-- outshape: {}".format(list(X.shapes)))
#     return X


@register_relay_2_xlayer_converter('transpose')
def transpose(expr: Expr,
              params: Dict[str, np.ndarray],
              schedule: Schedule,
              net: Dict[Expr, Expr],
              op_idx: Dict[str, int],
              RELAY_2_XLAYER: Dict[str, Callable],
              **kwargs) -> XLayer:
    """
    Relay Transpose to XLayer converter

    Relay
    -----
    Type: tvm.relay.op.transform.transpose
    Ref: https://docs.tvm.ai/api/python/relay/nn.html
    Parameters:
        - data (relay.Expr)
            The input data to the operator.
        - axes (None or List[int])
            The target axes order, reverse order if not specified.
    """
    if expr in net:
        logger.debug("MEMORY: TRANSPOSE")
        # This expressions is already transformed so we reuse that one
        return net[expr]

    expr_axes = expr.attrs.axes
    axes = [int(e) for e in list(expr_axes)] if expr_axes is not None else None

    data_expr, data_expr_class = expr.args[0], expr.args[0].__class__.__name__

    data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params, schedule,
                                                 net, op_idx, RELAY_2_XLAYER,
                                                 **kwargs)

    logger.debug("transpose")

    if 'Constant' in data_layer.type:

        logger.debug("-- constant")
        # TODO: TEST
        data = np.transpose(data_layer.data[0], tuple(axes))
        dtype = data_layer.attrs['dtype']

        op_name = 'constant-' + str(hash(expr))

        # Merge relay ids
        relay_idx = data_layer.attrs['relay_id'][:]
        relay_idx.append(hash(expr))

        X = xlf.get_xop_factory_func('Constant')(op_name,
                                                 data,
                                                 relay_id=relay_idx)
    else:
        # Update schedule with input data layer
        if data_expr not in net:
            schedule.append(data_expr)
            net[data_expr] = data_layer

        # Create XLayer
        # Relay converts a NHWC conv2d_transpose layer into a
        #   transpose -> conv2d_transpose (NCHW) -> transpose. For partitioning we
        #   keep track of those relay ids inside the conv2d_transpose operation
        if 'Conv2DTranspose' in data_layer.type:
            data_layer.attrs['relay_id'].append(hash(expr))

        # Create name
        op_name = 'transpose-' + str(hash(expr))

        X = xlf.get_xop_factory_func('Transpose')(op_name, data_layer,
                                                  axes,
                                                  relay_id=[hash(expr)])
        logger.debug("-- outshape: {}".format(list(X.shapes)))

        # !Important: set input layer tops:
        data_layer.tops.append(op_name)

    return X


@register_relay_2_xlayer_converter_base('zeros_like')
def zeros_like(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Zeros like to XLayer converter

    Relay
    -----
    Type: tvm.relay.zeros_like
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - data (relay.Expr)
            The input data
    """
    assert len(in_xlayers) == 1
    newshape = list(in_xlayers[0].shapes[:])
    X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])
    return X

