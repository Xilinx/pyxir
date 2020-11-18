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
Module for transforming Relay L5 operators to XLayer objects

L5: Vision operators
"""

import math
import logging
import warnings
import numpy as np
import pyxir as px

import tvm

from pyxir import graph
from pyxir.graph.layer import xlayer_factory as xlf

from .relay_2_xlayer_registry import register_relay_2_xlayer_converter,\
    register_relay_2_xlayer_converter_base

logger = logging.getLogger("pyxir")


@register_relay_2_xlayer_converter_base('nn.adaptive_avg_pool2d')
def nn_adaptive_avg_pool2d(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    Experimental 2D adaptive average pooling operator. Takes as
    argument the output size and automatically computes the kernel and
    strides.

    Relay
    -----
    Type: tvm.relay.nn.adaptive_avg_pool2d
    Ref: https://docs.tvm.ai/api/python/relay/nn.html
    Parameters:
        - data (tvm.relay.Expr)
            The input data to the operator.
        - output_size (tuple of int. optional)
            Output height and width.
        - layout (str, optional)
            Layout of the input.
    """
    assert len(in_xlayers) == 1
    warnings.warn("Convert Relay Adaptive Avg pool2d layer into normal"
                  " average pool2d layer")

    output_size = [int(e) for e in expr.attrs.output_size] \
        if expr.attrs.output_size is not None else None
    layout = str(expr.attrs.layout)

    h_idx, w_idx = layout.index('H'), layout.index('W')
    in_h = in_xlayers[0].shapes[h_idx]
    in_w = in_xlayers[0].shapes[w_idx]

    if output_size is None:
        out_h, out_w = in_h, in_w
    elif len(output_size) == 1:
        out_h, out_w = output_size[0], output_size[0]
    else:
        out_h, out_w = output_size

    stride_h, stride_w = in_h // out_h, in_w // out_w
    kernel_h = in_h - (out_h - 1) * stride_h
    kernel_w = in_w - (out_w - 1) * stride_w

    X = xlf.get_xop_factory_func('Pooling')(
        op_name=op_name,
        input_layer=in_xlayers[0],
        pool_type='Avg',
        pool_size=[kernel_h, kernel_w],
        strides=[stride_h, stride_w],
        padding=[0, 0],
        layout=layout,
        ceil_mode=False,
        count_include_pad=False,
        relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    return X


@register_relay_2_xlayer_converter_base('slice_like')
def slice_like(op_name, expr, in_xlayers):
    # type: (str, tvm.relay.expr.Expr, List[XLayer]) -> XLayer
    """
    Slice like

    Relay
    -----
    Type: tvm.relay.slice_like
    Ref: https://docs.tvm.ai/api/python/relay/index.html
    Parameters:
        - data (tvm.relay.Expr)
            The source array.
        - shape_like (tvm.relay.Expr)
            The new shape.
        - axes (Optional[Tuple[int]])
            List of axes on which input data will be sliced according to the
            corresponding size of the second input. By default will slice on
            all axes. Negative axes mean counting in reverse.
    """
    data_shapes = list(in_xlayers[0].shapes[:])
    shapes_like = list(in_xlayers[1].shapes[:])
    axes = [int(e) for e in list(expr.attrs.axes)] if expr.attrs.axes is not None\
        else list(range(min(len(data_shapes), len(shapes_like))))

    new_shape = data_shapes[:]
    for dim in axes:
        new_shape[dim] = shapes_like[dim]

    logger.debug("--newshape: {}".format(new_shape))

    X = px.ops.any_op(op_name, in_xlayers, any_shape=new_shape, relay_id=[hash(expr)])

    return X