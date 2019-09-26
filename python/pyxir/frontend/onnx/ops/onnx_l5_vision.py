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

L5: Vision operators


"""

import logging
import numpy as np
import pyxir as px

from typing import Dict

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from pyxir.shapes.tools import get_numpy_broadcasted_shape
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper, get_onnx_elem_type_2_dtype
from .tools import eltwise_any_op

logger = logging.getLogger('pyxir')


@register_onnx_2_xlayer_converter("NonMaxSuppression")
def non_max_suppression(node: NodeWrapper,
                        params: Dict[str, np.ndarray],
                        xmap: Dict[str, XLayer]):
    """ ONNX NonMaxSuppression to XLayer AnyOp conversion function """

    logger.info("ONNX NonMaxSupression -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    boxes_X = xmap[bottoms[0]]
    num_batches, spatial_d, _ = boxes_X.shapes.tolist()

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[boxes_X],
        any_shape=[num_batches, -1, 4],
        onnx_id=name
    )

    return [X]
