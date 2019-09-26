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
Utility module for transforming ONNX operators to XLayer objects


"""

import logging
import numpy as np
import pyxir as px

from typing import Dict

from pyxir.graph.layer import XLayer
from ..onnx_tools import NodeWrapper

logger = logging.getLogger("pyxir")


def eltwise_any_op(op_name: str,
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
