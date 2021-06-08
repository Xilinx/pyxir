

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

"""Module for transforming Relay operators to XLayer objects"""

import math
import logging
import numpy as np
import tvm
import pyxir as px

from typing import Dict, List, Callable
from tvm import relay
from tvm.relay.expr import Expr

from pyxir import graph
from pyxir.graph.layer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.shapes import TensorShape, TupleShape

from .util import Schedule
from .relay_2_xlayer_registry import register_relay_2_xlayer_converter,\
    register_relay_2_xlayer_converter_base

logger = logging.getLogger("pyxir")

#####################
# RELAY EXPRESSIONS #
#####################


@register_relay_2_xlayer_converter('Function')
def function(expr: Expr,
             params: Dict[str, np.ndarray],
             schedule: Schedule,
             net: Dict[Expr, Expr],
             op_idx: Dict[str, int],
             RELAY_2_XLAYER: Dict[str, Callable],
             **kwargs):
    """
    TVM function to XLayer converter

    Relay
    -----
    Type: tvm.relay.expr.Function
    Ref: https://docs.tvm.ai/api/python/relay/expr.html
    Parameters:
        - params (List[tvm.relay.Var])
            List of input parameters to the function.
        - body (tvm.relay.Expr)
            The body of the function.
        - ret_type (Optional[tvm.relay.Type])
            The return type annotation of the function.
        - type_params (Optional[List[tvm.relay.TypeParam]])
            The additional type parameters, this is only used in advanced
            usecase of template functions.
    """
    # TODO, do we need the params of the function?
    logger.debug(expr.body)
    body_class_name = expr.body.__class__.__name__
    X = RELAY_2_XLAYER[body_class_name](expr.body, params, schedule, net,
                                        op_idx, RELAY_2_XLAYER, **kwargs)

    # ! We still have to add the last layer to the network, because this
    #   layer doesn't have a parent operation that can decide whether to
    #   add or not add this layer
    if not expr.body in net:
        schedule.append(expr.body)
        net[expr.body] = X

    return X


@register_relay_2_xlayer_converter('Call')
def call(expr: Expr,
         params: Dict[str, np.ndarray],
         schedule: Schedule,
         net: Dict[Expr, Expr],
         op_idx: Dict[str, int],
         RELAY_2_XLAYER: Dict[str, Callable],
         **kwargs):
    """
    TVM Call to XLayer converter

    Relay
    -----
    Type: tvm.relay.expr.Call
    Ref: https://docs.tvm.ai/api/python/relay/expr.html
    Parameters:
        - op (tvm.relay.Op or any tvm.relay.Expr with function type.)
            The operation to be called.
        - args (List[tvm.relay.Expr])
            The arguments to the call.
        - attrs (Optional[tvm.Attrs])
            Attributes to the call, can be None
        - type_args (Optional[List[tvm.relay.Type]])
            The additional type arguments, this is only used in advanced
            usecase of template functions.
    """
    op_type = expr.op.name
    logger.debug("Call: {}".format(op_type))

    X = RELAY_2_XLAYER[op_type](expr, params, schedule, net, op_idx,
                                RELAY_2_XLAYER, **kwargs)

    return X


@register_relay_2_xlayer_converter_base('layout_transform')
def layout_transform(op_name: str, expr: Expr, in_xlayers: List[XLayer]):
    """
    TVM layout transform to XLayer converter

    Relay
    -----
    Type: tvm.relay.strided_slice
    Ref: https://docs.tvm.ai/api/python/relay/vision.html
    Parameters:
        - data (relay.Expr)
            The source tensor to be transformed
        - src_layout (str)
            The source layout. (e.g NCHW)
        - dst_layout (str)
            The destination layout. (e.g. NCHW16c)
    """
    inX = in_xlayers[0]
    in_shape = list(inX.shapes[:])
    src_layout = str(expr.attrs.src_layout)
    dst_layout = str(expr.attrs.dst_layout)
    assert len(src_layout) == len(dst_layout), "Layout transform source and destination layout"\
        " should be anagrams"

    transpose_axes = [src_layout.index(e) for e in dst_layout]
    if 'Constant' in inX.type:
        data = np.transpose(inX.data[0], transpose_axes)
        X = px.ops.constant(op_name, data, relay_id=[hash(expr)])
    else:
        newshape = [in_shape[i] for i in transpose_axes]
        X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter('Tuple')
def tuple_expr(expr: Expr,
               params: Dict[str, np.ndarray],
               schedule: Schedule,
               net: Dict[Expr, Expr],
               op_idx: Dict[str, int],
               RELAY_2_XLAYER: Dict[str, Callable],
               **kwargs):
    """
    TVM Tuple expression to XLayer converter

    Relay
    -----
    Tuple expression that groups several fields together.
    Type: tvm.relay.expr.Tuple
    Ref: https://docs.tvm.ai/api/python/relay/expr.html#tvm.relay.expr.Tuple
    Parameters:
        - fields (List[tvm.relay.Expr])
            The fields in the tuple.
    """

    data_layers = []
    for data_expr in expr.fields:
        data_expr_class = data_expr.__class__.__name__
        logger.debug("-- {}".format(data_expr_class))
        data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params,
                                                     schedule, net, op_idx,
                                                     RELAY_2_XLAYER, **kwargs)
        data_layers.append(data_layer)

        if data_expr not in net:
            net[data_expr] = data_layer
            schedule.append(data_expr)

    # Create tuple name
    op_name = 'tuple-' + str(hash(expr))
    logger.debug("Tuple: {}".format(op_name))

    X = px.ops.tuple(op_name, data_layers, relay_id=[hash(expr)])
    logger.debug("-- newshape: {}".format(list(X.shapes)))

    for data_layer in data_layers:
        data_layer.tops.append(X.name)

    return X


@register_relay_2_xlayer_converter('TupleGetItem')
def tuple_get_item(expr: Expr,
                   params: Dict[str, np.ndarray],
                   schedule: Schedule,
                   net: Dict[Expr, Expr],
                   op_idx: Dict[str, int],
                   RELAY_2_XLAYER: Dict[str, Callable],
                   **kwargs):
    """
    TVM TupleGetItem to XLayer converter

    Relay
    -----
    Type: tvm.relay.expr.TupleGetItem
    Ref: https://docs.tvm.ai/api/python/relay/expr.html?highlight=
         tuplegetitem#tvm.relay.expr.TupleGetItem
    Parameters:
        - tuple_value (tvm.relay.Expr)
            The input tuple expression.
        - index (int)
            The index.
    """
    if expr in net:
        logger.debug("MEMORY TupleGetItem: {}".format(str(hash(expr))))
        return net[expr]
    # TODO is this always correct?
    child_expr = expr.tuple_value
    child_expr_class = child_expr.__class__.__name__

    index = int(expr.index)
    logger.debug("TupleGetItem: {}".format(child_expr_class))
    logger.debug("-- index: {}".format(index))

    child_layer = RELAY_2_XLAYER[child_expr_class](child_expr, params,
                                                   schedule, net, op_idx,
                                                   RELAY_2_XLAYER, **kwargs)

    # TODO
    # For some operations, e.g. batch_norm we only use one of the elements of
    #   the tuple output. For now, this is taken care of here, we should handle
    #   this more generally in the future. E.g. what if the new running mean
    #   and variance is still used??
    # See https://docs.tvm.ai/langref/relay_op.html#tvm.relay.nn.batch_norm

    if isinstance(child_layer.shapes, TensorShape):
        # Skip TupleGetItem layer
        logger.debug("-- Skip this TGI for tensor child layer: {}".format(child_layer.name))

        X = child_layer
        X.attrs['relay_id'].append(hash(expr))
        if child_expr not in net:
            schedule.append(child_expr)
            net[child_expr] = X

        # Because we remove this tuple get item layer, we want it to refer to
        # the child layer in the net map
        net[expr] = X
    elif isinstance(child_layer.shapes, TupleShape):

        # Update schedule with input data layer
        if child_expr not in net:
            schedule.append(child_expr)
            net[child_expr] = child_layer

        # Create name
        op_name = 'tuple_get_item-' + str(hash(expr))

        X = xlf.get_xop_factory_func('TupleGetItem')(op_name, [child_layer],
                                                     index=index,
                                                     relay_id=[hash(expr)])

        child_layer.tops.append(X.name)
    else:
        raise ValueError("TupleGetItem layer has input layer with shape of"
                         " type: {}, but should be of type TupleShape or"
                         " TensorShape".format(type(child_layer.shapes)))

    return X


###################
# INPUT OPERATORS #
###################


@register_relay_2_xlayer_converter('Constant')
def constant(expr: Expr,
             params: Dict[str, np.ndarray],
             schedule: Schedule,
             net: Dict[Expr, Expr],
             op_idx: Dict[str, int],
             RELAY_2_XLAYER: Dict[str, Callable],
             **kwargs):
    """
    Relay Constant to XLayer converter

    Relay
    -----
    Type: tvm.relay.expr.const
    Ref: https://docs.tvm.ai/api/python/relay/expr.html
    Parameters:
        - value (Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray])
            The constant value.
        - dtype (str, optional)
            The data type of the value.
    Attributes:
        - data (Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray])
            The constant value
    """
    if expr in net:
        # raise ValueError("Relay constant expression should never be in"
        #                  " memory!")
        # This expressions is already transformed so we reuse that one
        logger.debug("MEMORY Constant: {}".format(str(hash(expr))))
        return net[expr]

    logger.debug("constant: {}".format(""))

    # Create XLayer
    value = expr.data
    if isinstance(value, tvm.nd.NDArray):
        value = value.asnumpy()
    if value.ndim == 0:
        value = value.reshape((-1,))

    # Create name
    op_name = 'constant-' + str(hash(expr))

    X = xlf.get_xop_factory_func('Constant')(op_name, value,
                                             relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter('Var')
def var(expr: Expr,
        params: Dict[str, np.ndarray],
        schedule: Schedule,
        net: Dict[Expr, Expr],
        op_idx: Dict[str, int],
        RELAY_2_XLAYER: Dict[str, Callable],
        **kwargs):
    """
    Relay Var to XLayer converter

    Relay
    -----
    Type: tvm.relay.expr.var
    Ref: https://docs.tvm.ai/api/python/relay/expr.html
    Parameters:
        - name_hint (str)
            The name of the variable. This name only acts as a hint,
            and is not used for equality.
        - type_annotation (Optional[tvm.relay.Type, str])
            The type annotation on the variable. When type_annotation is a
            str, we will create a scalar variable.
        - shape (Optional[List[tvm.Expr]])
            The shape of the tensor type.
        - dtype (str, optional)
            The data type of the tensor.
    """
    if expr in net:
        logger.debug("MEMORY: VAR")
        # This expressions is already transformed so we reuse that one
        return net[expr]
    logger.debug("var: {}".format(""))

    # Create XLayer
    name = expr.name_hint
    logger.debug("-- name: {}".format(name))
    shape = [ int(s.value) for s in list(expr.type_annotation.shape)]
    dtype = str(expr.type_annotation.dtype)
    logger.debug("-- shape: {}".format(shape))

    # Two possibilities, the name can be found in params dictionary or
    #   this variable will be provided at runtime
    if name in params:
        value = params[name]

        logger.debug("-- param: {}".format(name))

        if isinstance(value, tvm.nd.NDArray):
            value = value.asnumpy()
        elif not isinstance(value, np.ndarray):
            raise ValueError("Values in provided parameters dictionary should"
                             " be of type: `numpy.ndarray` or"
                             " `tvm.nd.NDArray` but found"
                             " type: {} for key: {}".format(type(value), name))

        X = px.ops.constant(name, value, relay_id=[hash(expr)])
    else:
        # data_layout = kwargs['data_layout']
        cvx_prep = kwargs['cvx_prep'] if 'cvx_prep' in kwargs else {}

        if name not in cvx_prep:
            X = px.ops.input(name, shape, dtype=dtype, relay_id=[hash(expr)])
        else:
            str_in_X = \
                xlf.get_xop_factory_func('StrInput')(name,
                                                     relay_id=[hash(expr)])

            op_name = name + '_cvx'

            str_in_X.tops.append(op_name)
            if str_in_X.name not in net:
                schedule.append(str_in_X.name)
                net[str_in_X.name] = str_in_X

            X = xlf.get_xop_factory_func('Cvx')(op_name, str_in_X,
                                                cvx_prep[name], shape,
                                                dtype,
                                                relay_id=[hash(expr)])

    return X


@register_relay_2_xlayer_converter_base('RelayOp')
def relay_op(op_name: str, expr: Expr, in_xlayers: List[XLayer]):
    """Insert generic RelayOp operator"""

    logger.debug("-- op_name: {}".format(op_name))
    logger.debug("-- expr: {}".format(expr.op))

    try:
        ty = expr.checked_type
    except ValueError as e:
        # TODO, this is not correct
        if expr.type_args and len(expr.type_args) > 0:
            ty = expr.type_args[0]
        else:
            raise e
        
    if isinstance(ty, relay.ty.TensorType):
        relay_shape = TensorShape([int(s.value) for s in list(ty.shape)])
        dtype = str(ty.dtype)
    else:
        relay_shape = TupleShape(
            [TensorShape([int(i) for i in list(t_ty.shape)])
             for t_ty in ty.fields])
        dtype = [str(t_ty.dtype) for t_ty in ty.fields]

    # TODO
    # relay_shape.set_value(axis=0, value=-1)

    attrs = {}
    for attr in dir(expr.attrs):
        value = getattr(expr.attrs, attr)
        attrs[attr] = str(value)

    if 'dtype' in attrs:
        dtype = attrs['dtype']
        del attrs['dtype']

    X = xlf.get_xop_factory_func('RelayOp')(op_name, in_xlayers,
                                            relay_shape=relay_shape.tolist(),
                                            dtype=dtype,
                                            relay_id=[hash(expr)],
                                            **attrs)

    return X
