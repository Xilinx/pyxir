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
Module for transforming ONNX L0 operators to XLayer objects

L0: Other, mostly input and graph utility operations


"""

import math
import logging
import numpy as np
import pyxir as px

from typing import Dict

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper
from .tools import eltwise_any_op

logger = logging.getLogger("pyxir")


@register_onnx_2_xlayer_converter('AnyOp')
def any_op(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX unknown op to XLayer UknownOp conversion function """

    logger.info("ONNX {} -> XLayer Unknown op"
                .format(node.get_op_type()))

    raise NotImplementedError("")


@register_onnx_2_xlayer_converter("Constant")
def constant(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    # print(node_attrs.keys())
    key = list(node_attrs.keys())[0]
    data = node_attrs[key]

    X = px.ops.constant(
        op_name=px.stringify(name),
        value=data,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ConstantOfShape")
def constant_of_shape(node: NodeWrapper,
                      params: Dict[str, np.ndarray],
                      xmap: Dict[str, XLayer]):
    raise NotImplementedError("")


@register_onnx_2_xlayer_converter("Identity")
def identity(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    return eltwise_any_op("Identity", node, params, xmap)


@register_onnx_2_xlayer_converter("If")
def if_op(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    raise NotImplementedError("ONNX If operator not supported in Pyxir")


@register_onnx_2_xlayer_converter("Loop")
def loop_op(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    raise NotImplementedError("ONNX Loop operator not supported in Pyxir")


@register_onnx_2_xlayer_converter("Range")
def range(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX Range to XLayer AnyOp conversion function """

    logger.info("ONNX Range -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    start = int(xmap[bottoms[0]].data[0])
    limit = int(xmap[bottoms[1]].data[0])
    delta = int(xmap[bottoms[2]].data[0])

    shape = [max(math.ceil((limit - start) / delta), 0)]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Shape")
def shape(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX Shape to XLayer AnyOp conversion function """

    logger.info("ONNX Shape -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    size = len(iX.shapes)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=[size],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Size")
def size(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX Size to XLayer AnyOp conversion function """

    logger.info("ONNX Size -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=[1],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("StringNormalizer")
def string_normalizer(node: NodeWrapper,
                      params: Dict[str, np.ndarray],
                      xmap: Dict[str, XLayer]):
    raise NotImplementedError("ONNX StringNormalizer operator not supported"
                              " in Pyxir")


@register_onnx_2_xlayer_converter("TopK")
def topk(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX TopK to XLayer AnyOp conversion function """

    logger.info("ONNX TopK -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    in_shape = iX.shapes.tolist()
    k = int(xmap[bottoms[1]].data[0])

    axis = node_attrs['axis'] if 'axis' in node_attrs else -1
    if axis < 0:
        axis += d

    out_shape = [(i if idx != axis else k) for idx, i in enumerate(in_shape)]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Unique")
def unique(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    raise NotImplementedError("ONNX Unique operator not supported in Pyxir")
