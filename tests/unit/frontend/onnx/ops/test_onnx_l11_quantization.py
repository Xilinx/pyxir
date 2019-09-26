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
Module for testing the L3 operators for the ONNX frontend


"""

import math
import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l11_quantization as ol11

from pyxir.shapes import TensorShape, TupleShape


class TestONNXL11Quantization(unittest.TestCase):

    def test_eltwise_any_ops(self):

        any_ops = ['DequantizeLinear', 'QuantizeLinear']

        for any_op in any_ops:
            a = np.zeros((1, 2, 3, 3), dtype=np.float32)

            node = onnx.helper.make_node(
                any_op,
                inputs=['a'],
                outputs=['y']
            )

            wrapped_node = NodeWrapper(node)

            aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                                   dtype='float32')

            xmap = {'a': aX}
            params = {}

            func = getattr(ol11, any_op.lower())
            Xs = func(wrapped_node, params, xmap)

            assert len(Xs) == 1
            X = Xs[0]

            assert X.name == 'y'
            assert 'AnyOp' in X.type
            assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_dynamic_quantize_linear(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'DynamicQuantizeLinear',
            inputs=['a'],
            outputs=['x', 'y', 'z']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol11.dynamic_quantize_linear(wrapped_node, params, xmap)

        assert len(Xs) == 4
        X = Xs[0]

        assert X.name == 'dql-x'
        assert 'AnyOp' in X.type
        assert X.shapes == TupleShape([TensorShape([-1, 2, 3, 3]),
                                       TensorShape([1]),
                                       TensorShape([1])])

        assert Xs[1].name == 'x'
        assert Xs[1].shapes == TensorShape([-1, 2, 3, 3])
        assert Xs[2].name == 'y'
        assert 'TupleGetItem' in Xs[2].type
        assert Xs[2].shapes == TensorShape([1])
        assert Xs[3].name == 'z'
        assert 'TupleGetItem' in Xs[3].type
        assert Xs[3].shapes == TensorShape([1])
