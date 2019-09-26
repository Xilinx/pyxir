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

import tvm

from pyxir import graph
from pyxir.graph.layer import xlayer_factory as xlf

from .relay_2_xlayer_registry import register_relay_2_xlayer_converter

logger = logging.getLogger("pyxir")


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
