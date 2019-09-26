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
Module for testing the L5 operators for the ONNX frontend


"""

import math
import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l5_vision as ol5

from pyxir.shapes import TensorShape, TupleShape


class TestONNXL5Vision(unittest.TestCase):

    def test_non_max_suppression(self):

        a = np.zeros((1, 3, 4), dtype=np.float32)

        node = onnx.helper.make_node(
            'NonMaxSuppression',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol5.non_max_suppression(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, -1, 4]
