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
import numpy as np

import tvm

from pyxir import graph
from pyxir.graph.layer import xlayer_factory as xlf

from .relay_2_xlayer_registry import register_relay_2_xlayer_converter,\
    register_relay_2_xlayer_converter_base

logger = logging.getLogger("pyxir")


@register_relay_2_xlayer_converter('vision.yolo_reorg')
def yolo_reorg(expr, params, schedule, net, op_idx, RELAY_2_XLAYER, **kwargs):
    # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
    #   Dict[int, XLayer], Dict[str, int], Dict[str, Function]) -> XLayer
    """
    Conversion of Relay 'yolo_reorg' layer

    Relay
    -----
    Type: tvm.relay.vision.yolo_reorg
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (relay.Expr)
            The input data tensor.
        - stride (int)
            The stride value for reorganisation.
    """
    if expr in net:
        # This expressions is already transformed so we reuse that one
        return net[expr]

    stride = int(expr.attrs.stride)

    data_expr, data_expr_class = expr.args[0], expr.args[0].__class__.__name__

    data_layer = RELAY_2_XLAYER[data_expr_class](data_expr, params, schedule,
                                                 net, op_idx, RELAY_2_XLAYER,
                                                 **kwargs)

    logger.debug("yolo reorg:")

    # Update schedule with input data layer
    if data_expr not in net:
        schedule.append(data_expr)
        net[data_expr] = data_layer

    # Create XLayer
    op_name = 'yolo_reorg-' + str(hash(expr))

    X = xlf.get_xop_factory_func('YoloReorg')(op_name, data_layer,
                                              stride, 'NCHW',
                                              relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))

    # !Important: set input layer tops:
    data_layer.tops.append(op_name)

    return X
