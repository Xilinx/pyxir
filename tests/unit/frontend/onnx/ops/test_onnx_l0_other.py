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
Module for testing the L0 operators for the ONNX frontend


"""

import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l0_other as ol0


class TestONNXL0Other(unittest.TestCase):

    def test_identity(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Identity',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol0.identity(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_range(self):
        start = np.array([1])
        limit = np.array([10])
        delta = np.array([2])

        node = onnx.helper.make_node(
            'Range',
            inputs=['s', 'l', 'd'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        sX = xlf.get_xop_factory_func('Constant')('s', start)
        lX = xlf.get_xop_factory_func('Constant')('l', limit)
        dX = xlf.get_xop_factory_func('Constant')('d', delta)

        xmap = {'s': sX, 'l': lX, 'd': dX}
        params = {}

        Xs = ol0.range(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [5]

    def test_shape(self):
        a = np.zeros((1, 2, 3, 3))

        node = onnx.helper.make_node(
            'Shape',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))

        xmap = {'a': aX}
        params = {}

        Xs = ol0.shape(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [4]

    def test_size(self):
        a = np.zeros((1, 2, 3, 3))

        node = onnx.helper.make_node(
            'Size',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))

        xmap = {'a': aX}
        params = {}

        Xs = ol0.size(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [1]

    def test_top_k(self):
        a = np.zeros((1, 4, 3, 3))
        k = np.array([2])

        node = onnx.helper.make_node(
            'TopK',
            inputs=['a', 'k'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))
        kX = xlf.get_xop_factory_func('Constant')('k', k)

        xmap = {'a': aX, 'k': kX}
        params = {}

        Xs = ol0.topk(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
