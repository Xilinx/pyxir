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
Module for testing the L4 operators for the ONNX frontend


"""

import math
import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l6_recurrent as ol6

from pyxir.shapes import TensorShape, TupleShape


class TestONNXL6Recurrent(unittest.TestCase):
    pass
