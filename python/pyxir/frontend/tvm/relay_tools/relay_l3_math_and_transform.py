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

import tvm

from pyxir import graph
from pyxir.graph.layer import xlayer_factory as xlf

from .relay_2_xlayer_registry import register_relay_2_xlayer_converter,\
    register_relay_2_xlayer_converter_base

logger = logging.getLogger("pyxir")


@register_relay_2_xlayer_converter_base('cast')
def cast(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
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


@register_relay_2_xlayer_converter('clip')
def clip(expr, params, schedule, net, op_idx, RELAY_2_XLAYER, **kwargs):
    # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
    #   Dict[int, XLayer], Dict[str, int], Dict[str, Function]) -> XLayer
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
    if expr in net:
        # This expressions is already transformed so we reuse that one
        return net[expr]

    a_min = float(expr.attrs.a_min)
    a_max = float(expr.attrs.a_max)

    data_expr, data_expr_class = expr.args[0], expr.args[0].__class__.__name__

    data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params, schedule,
                                                 net, op_idx, RELAY_2_XLAYER,
                                                 **kwargs)

    logger.debug("clip: {}".format(""))

    # Update schedule with input data layer
    if data_expr not in net:
        schedule.append(data_expr)
        net[data_expr] = data_layer

    # Create XLayer
    op_name = 'clip-' + str(hash(expr))

    X = xlf.get_xop_factory_func('Clip')(op_name, data_layer,
                                         a_min, a_max,
                                         relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    # !Important: set input layer tops:
    data_layer.tops.append(op_name)

    return X


@register_relay_2_xlayer_converter_base('nn.leaky_relu')
def nn_leaky_relu(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    Compute leaky rectified linear unit nonlinearity

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

    X = xlf.get_xop_factory_func('LeakyReLU')(op_name, in_xlayers,
                                              alpha=alpha,
                                              relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter_base('nn.prelu')
def nn_prelu(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
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

    alpha = float(expr.attrs.alpha)
    axis = int(expr.attrs.axis) if expr.attrs.axis is not None else 1

    X = xlf.get_xop_factory_func('pReLU')(op_name,
                                          in_xlayers[0],
                                          alpha,
                                          axis,
                                          relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter('reshape')
def reshape(expr, params, schedule, net, op_idx, RELAY_2_XLAYER, **kwargs):
    # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
    #   Dict[int, XLayer], Dict[str, int], Dict[str, Function]) -> XLayer
    """
    TODO

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

    logger.debug("reshape: {}".format(""))
    logger.debug("relay shape: {}".format(relayshape))
    # Parse the Relay newshape list because it can contain special numbers
    # (https://docs.tvm.ai/api/python/relay/op.html#tvm.relay.op.transform.reshape)
    # TODO TEST
    input_shape = list(data_layer.shapes)

    newshape = []
    i, j = 0, 0  # i is index in relayshape, j in input_shape
    while i < len(relayshape):
        dim = relayshape[i]
        if dim > 0:
            newshape.append(dim)
        elif dim == 0:
            newshape.append(input_shape[i])
        elif dim == -1 and i == 0:
            newshape.append(-1)
        elif dim == -1 and i > 0:
            newshape.append(int(np.prod(input_shape[j:])))
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
            newshape.extend([nxt, nxtnxt])
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
        assert np.prod(list(data_layer.shapes)) == np.prod(newshape)

    # Create XLayer
    # Create name
    op_name = 'reshape-' + str(hash(expr))

    X = xlf.get_xop_factory_func('Reshape')(op_name, data_layer,
                                            newshape,
                                            relay_id=[hash(expr)])

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
def split(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
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

    X = xlf.get_xop_factory_func('Split')(op_name, in_xlayers,
                                          axis=axis,
                                          indices=indices,
                                          relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    return X


@register_relay_2_xlayer_converter('squeeze')
def squeeze(expr, params, schedule, net, op_idx, RELAY_2_XLAYER, **kwargs):
    # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
    #   Dict[int, XLayer], Dict[str, int], Dict[str, Function]) -> XLayer
    """
    TODO

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
    if expr in net:
        logger.debug("MEMORY: SQUEEZE")
        # This expressions is already transformed so we reuse that one
        return net[expr]

    expr_axis = expr.attrs.axis
    axis = [int(e) for e in list(expr_axis)] if expr_axis is not None else None

    data_expr, data_expr_class = expr.args[0], expr.args[0].__class__.__name__

    data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params, schedule,
                                                 net, op_idx, RELAY_2_XLAYER,
                                                 **kwargs)

    logger.debug("squeeze: {}".format(""))

    # Update schedule with input data layer
    if data_expr not in net:
        schedule.append(data_expr)
        net[data_expr] = data_layer

    # Create XLayer
    op_name = 'squeeze-' + str(hash(expr))

    X = xlf.get_xop_factory_func('Squeeze')(op_name, data_layer, axis,
                                            relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    # !Important: set input layer tops:
    data_layer.tops.append(op_name)

    return X


@register_relay_2_xlayer_converter_base('take')
def take(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    TODO

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

    X = xlf.get_xop_factory_func('Take')(op_name, in_xlayers,
                                         axis=axis,
                                         mode=mode,
                                         relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    return X


@register_relay_2_xlayer_converter('transpose')
def transpose(expr, params, schedule, net, op_idx, RELAY_2_XLAYER, **kwargs):
    # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
    #   Dict[int, XLayer], Dict[str, int], Dict[str, Function]) -> XLayer
    """
    TODO

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

        X = xlf.get_xop_factory_func('Constant')(op_name,
                                                 data,
                                                 relay_id=[hash(expr)])
    else:
        # Update schedule with input data layer
        if data_expr not in net:
            schedule.append(data_expr)
            net[data_expr] = data_layer

        # Create XLayer
        # data_layout = kwargs['data_layout']

        # Create name
        op_name = 'transpose-' + str(hash(expr))

        X = xlf.get_xop_factory_func('Transpose')(op_name, data_layer,
                                                  axes,
                                                  relay_id=[hash(expr)])
        logger.debug("-- outshape: {}".format(list(X.shapes)))

        # !Important: set input layer tops:
        data_layer.tops.append(op_name)

    return X
