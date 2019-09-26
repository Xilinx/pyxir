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
Module for transforming ONNX operators to XLayer objects

L11: Quantization related operators


"""

import logging
import numpy as np
import pyxir as px

from typing import Dict

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from pyxir.shapes.tools import get_numpy_broadcasted_shape
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper

logger = logging.getLogger('pyxir')


def eltwise_any_op(op_name,
                   node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX Eltwise op to XLayer AnyOp conversion function """

    logger.info("ONNX -> XLayer AnyOp".format(op_name))

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=iX.shapes.tolist(),
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("DequantizeLinear")
def dequantizelinear(node: NodeWrapper,
                     params: Dict[str, np.ndarray],
                     xmap: Dict[str, XLayer]):
    return eltwise_any_op("DequantizeLinear", node, params, xmap)


@register_onnx_2_xlayer_converter("DynamicQuantizeLinear")
def dynamic_quantize_linear(node: NodeWrapper,
                            params: Dict[str, np.ndarray],
                            xmap: Dict[str, XLayer]):
    """ ONNX DynamicQuantizeLinear to XLayer AnyOp conversion function """

    logger.info("ONNX DynamicQuantizeLinear -> XLayer AnyOp")

    # TODO first name is used for split for now
    name = node.get_outputs()[0]
    nb_outputs = len(node.get_outputs())
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)

    Xs = []
    X = px.ops.any_op(
        op_name='dql-' + px.stringify(name),
        in_xlayers=[iX],
        any_shape=[iX.shapes.tolist(), [1], [1]],
        onnx_id=name
    )
    Xs.append(X)

    for idx, tensor_name in enumerate(node.get_outputs()):
        tgi_X = px.ops.tuple_get_item(
            op_name=px.stringify(tensor_name),
            in_xlayers=[X],
            index=idx,
            onnx_id=tensor_name
        )
        Xs.append(tgi_X)

    return Xs


@register_onnx_2_xlayer_converter("QuantizeLinear")
def quantizelinear(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    return eltwise_any_op("QuantizeLinear", node, params, xmap)
