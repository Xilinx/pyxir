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
Module for transforming ONNX L3 operators to XLayer objects

L3: Additional math and transform operators


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

logger = logging.getLogger('pyxir')


@register_onnx_2_xlayer_converter("GRU")
def gru(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX GRU to XLayer AnyOp conversion function """

    logger.info("ONNX GRU -> XLayer AnyOp")

    raise NotImplementedError("ONNX GRU operator conversion not supported")


@register_onnx_2_xlayer_converter("LSTM")
def lstm(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX LSTM to XLayer AnyOp conversion function """

    logger.info("ONNX LSTM -> XLayer AnyOp")

    raise NotImplementedError("ONNX LSTM operator conversion not supported")


@register_onnx_2_xlayer_converter("RNN")
def rnn(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX RNN to XLayer AnyOp conversion function """

    logger.info("ONNX RNN -> XLayer AnyOp")

    raise NotImplementedError("ONNX RNN operator conversion not supported")


@register_onnx_2_xlayer_converter("Scan")
def scan(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX RNN to XLayer AnyOp conversion function """

    logger.info("ONNX Scan -> XLayer AnyOp")

    raise NotImplementedError("ONNX Scan operator conversion not supported")
