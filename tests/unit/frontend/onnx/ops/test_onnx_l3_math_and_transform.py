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
from pyxir.frontend.onnx.ops import onnx_l3_math_and_transform as ol3

from pyxir.shapes import TensorShape, TupleShape


class TestONNXL3MathAndTransform(unittest.TestCase):

    def test_eltwise_any_ops(self):

        any_ops = ['Abs', 'Acos', 'Acosh', 'Asin', 'Asinh', 'Atan', 'Atanh',
                   'BitShift', 'Ceil', 'Celu', 'Clip', 'Cos', 'Cosh',
                   'CumSum', 'Elu', 'Erf', 'EyeLike', 'Floor', 'HardSigmoid',
                   'IsInf', 'IsNaN', 'LpNormalization',
                   'MeanVarianceNormalization', 'Neg', 'NonZero',
                   'RandomNormalLike', 'RandomUniformLike', 'ReverseSequence',
                   'Round', 'Selu', 'Shrink', 'Sign', 'Sin', 'Sinh',
                   'SoftPlus', 'SoftSign', 'Tan', 'ThresholdedRelu'
                   ]

        any_op_to_func_name = {
            'Abs': 'abs_op'
        }

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

            if any_op not in any_op_to_func_name:
                func = getattr(ol3, any_op.lower())
            else:
                func = getattr(ol3, any_op_to_func_name[any_op].lower())
            Xs = func(wrapped_node, params, xmap)

            assert len(Xs) == 1
            X = Xs[0]

            assert X.name == 'y'
            assert 'AnyOp' in X.type
            assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_and(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'And',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol3.and_op(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_cast_float32(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Cast',
            inputs=['a'],
            outputs=['y'],
            to=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.cast(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Cast' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['dtype'] == 'float32'

    def test_cast_int32(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Cast',
            inputs=['a'],
            outputs=['y'],
            to=6
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.cast(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Cast' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['dtype'] == 'int32'

    def test_determinant(self):
        a = np.zeros((2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Det',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)

        xmap = {'a': aX}
        params = {}

        Xs = ol3.det(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [2]

    def test_hard_max(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Hardmax',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))

        xmap = {'a': aX}
        params = {}

        Xs = ol3.hard_max(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 18]

    def test_leaky_relu(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'LeakyRelu',
            inputs=['a'],
            outputs=['y'],
            alpha=0.2
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.leaky_relu(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'LeakyReLU' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert math.isclose(X.attrs['alpha'], 0.2, rel_tol=1e-5)

    def test_leaky_relu_default(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'LeakyRelu',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.leaky_relu(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'LeakyReLU' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['alpha'] == 0.01

    def test_multinomial(self):
        a = np.zeros((2, 2), dtype=np.float32)

        node = onnx.helper.make_node(
            'Multinomial',
            inputs=['a'],
            outputs=['y'],
            sample_size=10
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))

        xmap = {'a': aX}
        params = {}

        Xs = ol3.multinomial(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 10]

    def test_negative_log_likelihood_loss(self):
        a = np.zeros((2, 3, 2), dtype=np.float32)
        b = np.array([[2, 1], [0, 2]])

        node = onnx.helper.make_node(
            'NegativeLogLikelihoodLoss',
            inputs=['a', 'b'],
            outputs=['y'],
            reduction='none'
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol3.negative_log_likelihood_loss(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [2, 2]

    def test_one_hot(self):
        a = np.zeros((1, 3), dtype=np.float32)
        d = np.array([10])

        node = onnx.helper.make_node(
            'OneHot',
            inputs=['a', 'd'],
            outputs=['y'],
            reduction='none'
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))
        dX = xlf.get_xop_factory_func('Constant')('d', d)

        xmap = {'a': aX, 'd': dX}
        params = {}

        Xs = ol3.one_hot(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 3, 10]

    def test_or(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Or',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol3.or_op(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_prelu(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'PRelu',
            inputs=['a'],
            outputs=['y'],
            slope=0.2
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.prelu(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'LeakyReLU' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert math.isclose(X.attrs['alpha'], 0.2, rel_tol=1e-5)

    def test_pow(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Pow',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol3.pow(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_random_normal(self):

        node = onnx.helper.make_node(
            'RandomNormal',
            inputs=[],
            outputs=['y'],
            shape=[1, 3]
        )

        wrapped_node = NodeWrapper(node)

        xmap = {}
        params = {}

        Xs = ol3.random_normal(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [1, 3]

    def test_random_uniform(self):

        node = onnx.helper.make_node(
            'RandomUniform',
            inputs=[],
            outputs=['y'],
            shape=[1, 3]
        )

        wrapped_node = NodeWrapper(node)

        xmap = {}
        params = {}

        Xs = ol3.random_uniform(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [1, 3]

    def test_softmax_cross_entropy_loss(self):
        a = np.zeros((2, 3, 2), dtype=np.float32)
        b = np.array([[2, 1], [0, 2]])

        node = onnx.helper.make_node(
            'SoftmaxCrossEntropyLoss',
            inputs=['a', 'b'],
            outputs=['y'],
            reduction='none'
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol3.softmax_cross_entropy_loss(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [2, 2]

    def test_split(self):
        a = np.zeros((1, 5, 4, 4), dtype=np.float32)

        node = onnx.helper.make_node(
            'Split',
            inputs=['a'],
            outputs=['x', 'y', 'z'],
            axis=1,
            split=[1, 3, 1]
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.split(wrapped_node, params, xmap)

        assert len(Xs) == 4
        X = Xs[0]

        assert X.name == 'split-x'
        assert 'Split' in X.type
        assert X.attrs['axis'] == 1
        assert X.attrs['indices'] == [1, 4]
        assert X.shapes == TupleShape([TensorShape([-1, 1, 4, 4]),
                                       TensorShape([-1, 3, 4, 4]),
                                       TensorShape([-1, 1, 4, 4])])

        assert Xs[1].name == 'x'
        assert Xs[2].name == 'y'
        assert Xs[3].name == 'z'

    def test_split_default(self):
        a = np.zeros((1, 6), dtype=np.float32)

        node = onnx.helper.make_node(
            'Split',
            inputs=['a'],
            outputs=['x', 'y', 'z'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.split(wrapped_node, params, xmap)

        assert len(Xs) == 4
        X = Xs[0]

        assert X.name == 'split-x'
        assert 'Split' in X.type
        assert X.attrs['axis'] == 1
        assert X.attrs['indices'] == [2, 4]
        assert X.shapes == TupleShape([TensorShape([-1, 2]),
                                       TensorShape([-1, 2]),
                                       TensorShape([-1, 2])])

        assert Xs[1].name == 'x'
        assert Xs[2].name == 'y'
        assert Xs[3].name == 'z'

    def test_squeeze(self):
        a = np.zeros((1, 1, 1, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Squeeze',
            inputs=['a'],
            outputs=['y'],
            axes=[1, 2]
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))

        xmap = {'a': aX}
        params = {}

        Xs = ol3.squeeze(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Squeeze' in X.type
        assert X.shapes.tolist() == [-1, 3, 3]

    def test_tile(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        repeats = np.array([1, 3, 2, 2])

        node = onnx.helper.make_node(
            'Tile',
            inputs=['a', 'repeats'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))
        rX = xlf.get_xop_factory_func('Constant')('repeats', repeats)

        xmap = {'a': aX, 'repeats': rX}
        params = {}

        Xs = ol3.tile(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 6, 6, 6]

    def test_transpose(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Transpose',
            inputs=['a'],
            outputs=['y'],
            perm=[0, 2, 3, 1]
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.transpose(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Transpose' in X.type
        assert X.shapes.tolist() == [-1, 3, 3, 2]

    def test_transpose_default(self):
        # Unspecified transpose reverses the order
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Transpose',
            inputs=['a'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol3.transpose(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Transpose' in X.type
        assert X.shapes.tolist() == [3, 3, 2, -1]

    def test_unsqueeze(self):
        a = np.zeros((1, 3, 4, 5), dtype=np.float32)

        node = onnx.helper.make_node(
            'UnSqueeze',
            inputs=['a'],
            outputs=['y'],
            axes=[1, 5]
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape))

        xmap = {'a': aX}
        params = {}

        Xs = ol3.un_squeeze(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 4, 5, 1]

    def test_xor(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Xor',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol3.xor(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]
