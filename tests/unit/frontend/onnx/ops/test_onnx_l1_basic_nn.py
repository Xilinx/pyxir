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

"""Module for testing the L1 operators for the ONNX frontend"""

import math
import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l1_basic_nn as ol1b


class TestONNXL1BasicNN(unittest.TestCase):

    def test_eltwise_any_ops(self):

        any_ops = ['InstanceNormalization', 'LogSoftmax', 'Reciprocal']

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

            func = getattr(ol1b, any_op.lower())
            Xs = func(wrapped_node, params, xmap)

            assert len(Xs) == 1
            X = Xs[0]

            assert X.name == 'y'
            assert 'AnyOp' in X.type
            assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_add_two_constants(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Add',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.add(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Constant' in X.type
        assert X.shapes.tolist() == [1, 1, 3, 3]
        np.testing.assert_array_equal(
            X.data[0], np.zeros((1, 1, 3, 3), dtype=np.float32))

    def test_add_constant_tensor(self):
        a = np.array([1]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Add',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.add(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'BiasAdd' in X.type
        assert X.bottoms == ['b']
        np.testing.assert_array_equal(X.data[0], a)
        assert X.shapes.tolist() == [-1, 1, 3, 3]
        assert X.attrs['axis'] == 1

    def test_add_tensor_constant(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Add',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.add(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X = Xs[-1]

        assert X.name == 'y'
        assert 'Add' in X.type
        assert X.bottoms == ['a', 'b']
        # np.testing.assert_array_equal(X.data[0], b)
        assert X.shapes.tolist() == [1, 1, 3, 3]

    def test_add_eltwise(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Add',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.add(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Eltwise' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_batchnorm(self):
        x = np.array([[[[1, 2, 3]],
                       [[4, 5, 6]]]]).astype(np.float32)
        s = np.array([1, 2]).astype(np.float32)
        bias = np.array([-1, 1]).astype(np.float32)
        mean = np.array([0, 1]).astype(np.float32)
        var = np.array([1, 2]).astype(np.float32)

        node = onnx.helper.make_node(
            'BatchNormalization',
            inputs=['x', 's', 'bias', 'mean', 'var'],
            outputs=['y'],
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        sX = xlf.get_xop_factory_func('Constant')('s', s, onnx_id='s')
        bX = xlf.get_xop_factory_func('Constant')('bias', bias,
                                                  onnx_id='bias')
        mX = xlf.get_xop_factory_func('Constant')('mean', mean,
                                                  onnx_id='mean')
        vX = xlf.get_xop_factory_func('Constant')('var', var,
                                                  onnx_id='var')

        xmap = {'x': iX, 's': sX, 'bias': bX, 'mean': mX, 'var': vX}
        params = {}

        Xs = ol1b.batchnorm(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'BatchNorm' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 2, 1, 3]
        assert X.attrs['axis'] == 1
        assert X.attrs['epsilon'] == 1e-05

    def test_concat(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Concat',
            inputs=['a', 'b'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.concat(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Concat' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['axis'] == 1

    def test_concat_negative_axis(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Concat',
            inputs=['a', 'b'],
            outputs=['y'],
            axis=-3
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.concat(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Concat' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['axis'] == 1

    def test_divide(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Div',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.div(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Divide' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_dropout(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Dropout',
            inputs=['x'],
            outputs=['y'],
            ratio=0.7
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.dropout(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Dropout' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]
        assert math.isclose(X.attrs['rate'], 0.7, rel_tol=1e-5)

    def test_exp(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Exp',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.exp(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Exp' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_gemm_basic(self):
        x = np.array([[1, 2]]).astype(np.float32)  # 1 x 2
        w = np.array([[1, 2],
                      [3, 4],
                      [5, 6]]).astype(np.float32)  # 3 x 2
        b = np.array([-1, 1, 1]).astype(np.float32)

        node = onnx.helper.make_node(
            'Gemm',
            inputs=['x', 'w', 'b'],
            outputs=['y'],
            transB=1
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        wX = xlf.get_xop_factory_func('Constant')('w', w, onnx_id='w')
        bX = xlf.get_xop_factory_func('Constant')('b', b, onnx_id='b')

        xmap = {'x': iX, 'w': wX, 'b': bX}
        params = {}

        Xs = ol1b.gemm(wrapped_node, params, xmap)

        assert len(Xs) == 2
        dX, X = Xs

        assert X.name == 'y'
        assert 'BiasAdd' in X.type
        assert X.bottoms == ['y_Dense']
        assert X.shapes.tolist() == [-1, 3]

        assert dX.name == 'y_Dense'
        assert dX.shapes == [-1, 3]
        assert dX.bottoms == ['x']
        assert dX.attrs['units'] == 3
        assert dX.attrs['W_shape'] == [3, 2]

    def test_log(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Log',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.log(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Log' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_matmul_basic(self):
        x = np.array([[1, 2]]).astype(np.float32)  # 1 x 2
        w = np.transpose(np.array([
            [1, 2],
            [3, 4],
            [5, 6]]).astype(np.float32))  # 2 x 3

        node = onnx.helper.make_node(
            'MatMul',
            inputs=['x', 'w'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        wX = xlf.get_xop_factory_func('Constant')('w', w, onnx_id='w')

        xmap = {'x': iX, 'w': wX}
        params = {}

        Xs = ol1b.matmul(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Dense' in X.type
        assert X.shapes.tolist() == [-1, 3]

    def test_matmul_integer_basic(self):
        x = np.array([[1, 2]]).astype(np.int32)  # 1 x 2
        w = np.transpose(np.array([
            [1, 2],
            [3, 4],
            [5, 6]]).astype(np.int32))  # 2 x 3

        node = onnx.helper.make_node(
            'MatMulInteger',
            inputs=['x', 'w'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='int32')
        wX = xlf.get_xop_factory_func('Constant')('w', w, onnx_id='w')

        xmap = {'x': iX, 'w': wX}
        params = {}

        Xs = ol1b.matmul(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Dense' in X.type
        assert X.shapes.tolist() == [-1, 3]

    def test_mod(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Mod',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.mod(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_multiply_two_constants(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Mul',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.mul(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Constant' in X.type
        assert X.shapes.tolist() == [1, 1, 3, 3]
        res = np.array([[[[-1, -4, -9],
                          [-16, -25, -36],
                          [-49, -64, -81]]]]).astype(np.float32)
        np.testing.assert_array_equal(X.data[0], res)

    def test_mul_constant_tensor(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Mul',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.mul(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X = Xs[-1]

        assert X.name == 'y'
        assert 'Scale' in X.type
        assert X.bottoms == ['b']
        np.testing.assert_array_equal(X.data.gamma, a)
        assert X.shapes.tolist() == [-1, 1, 3, 3]
        assert X.attrs['axis'] == -1

    def test_mul_tensor_constant(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Mul',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.mul(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X = Xs[-1]

        assert X.name == 'y'
        assert 'Scale' in X.type
        assert X.bottoms == ['a']
        np.testing.assert_array_equal(X.data.gamma, b)
        assert X.shapes.tolist() == [-1, 1, 3, 3]
        assert X.attrs['axis'] == -1

    def test_multiply(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Mul',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.mul(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Multiply' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_qlinear_matmul(self):
        a = np.array([[1, 2]]).astype(np.float32)  # 1 x 2
        b = np.transpose(np.array([
            [1, 2],
            [3, 4],
            [5, 6]]).astype(np.float32))  # 2 x 3

        node = onnx.helper.make_node(
            'QLinearMatMul',
            inputs=['a', 'a_scale', 'a_zero', 'b', 'b_scale', 'b_zero',
                    'y_scale', 'y_zero'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b, onnx_id='b')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.qlinear_matmul(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 3]

    def test_relu(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Relu',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.relu(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'ReLU' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_sigmoid(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Sigmoid',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.sigmoid(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Sigmoid' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_softmax(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Softmax',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.softmax(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Softmax' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_sqrt(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Sqrt',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.sqrt(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Sqrt' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_sub_two_constants(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Sub',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.sub(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Constant' in X.type
        assert X.shapes.tolist() == [1, 1, 3, 3]
        np.testing.assert_array_equal(
            X.data[0], np.zeros((1, 1, 3, 3), dtype=np.float32))

    def test_sub(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Sub',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol1b.sub(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Sub' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_sum(self):
        a = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        b = np.array([[[[-1, -2, -3],
                        [-4, -5, -6],
                        [-7, -8, -9]]]]).astype(np.float32)

        c = np.zeros((1, 1, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Sum',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Input')('b', list(b.shape),
                                               dtype='float32')
        cX = xlf.get_xop_factory_func('Input')('c', list(c.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX, 'c': cX}
        params = {}

        Xs = ol1b.sum(wrapped_node, params, xmap)

        assert len(Xs) == 2

        X = Xs[0]
        assert X.name == 'y'
        assert 'Add' in X.type
        assert X.bottoms == ['a', 'b']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

        X = Xs[1]
        assert X.name == 'y2'
        assert 'Add' in X.type
        assert X.bottoms == ['y', 'c']
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_tanh(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Tanh',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol1b.tanh(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Tanh' in X.type
        assert X.bottoms == ['x']
        assert X.shapes.tolist() == [-1, 1, 3, 3]
