#!/usr/bin/env python
#
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

"""Module for testing the pyxir TF executor"""

import unittest
import numpy as np
import pyxir as px

from pyxir.shapes import TensorShape
from pyxir.runtime import base
from pyxir.graph.layer import xlayer
from pyxir.graph.io import xlayer_io

try:
    from pyxir.runtime.tensorflow.x_2_tf_registry import X_2_TF
    from pyxir.runtime.tensorflow.ops.tf_l0_input_and_other import *
    from pyxir.runtime.tensorflow.ops.tf_l1_basic_nn import *
    skip_tf = False
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Tensorflow related test because Tensorflow is not available")



class TestRuntimeTF(unittest.TestCase):

    def test_add_same_shape(self):
        left = np.array([[[[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]],
                          [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]]], dtype=np.float32)
        right = np.array([[[[-1, -2, -3],
                            [-4, -5, -6],
                            [-7, -8, -9]],
                           [[-1, -1, -1],
                            [-1, -1, -1],
                            [-1, -1, -1]]]], dtype=np.float32)

        X = xlayer.XLayer(
            name='add',
            type=['Add'],
            shapes=[1, 2, 3, 3],
            sizes=[18],
            bottoms=['left', 'right'],
            tops=[],
            attrs={},
            targets=[]
        )

        input_shapes = {
            'left': TensorShape([1, 2, 3, 3]),
            'right': TensorShape([1, 2, 3, 3])
        }
        inputs = {
            'left': left,
            'right': right
        }
        layers = X_2_TF['Add'](X, input_shapes, {})
        assert len(layers) == 1

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

        expected_outpt = np.zeros((1, 2, 3, 3), dtype=np.float32)

        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_add_broadcast(self):
        left = np.array([[[[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]],
                          [[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]]], dtype=np.float32)
        right = np.array([-1, -1, -1], dtype=np.float32)

        X = xlayer.XLayer(
            name='add',
            type=['Add'],
            shapes=[1, 2, 3, 3],
            sizes=[18],
            bottoms=['left', 'right'],
            tops=[],
            attrs={},
            targets=[]
        )

        input_shapes = {
            'left': TensorShape([1, 2, 3, 3]),
            'right': TensorShape([3])
        }
        inputs = {
            'left': left,
            'right': right
        }
        layers = X_2_TF['Add'](X, input_shapes, {})
        assert len(layers) == 1

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

        expected_outpt = \
            np.array([[[[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]]], dtype=np.float32)

        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_bias_add(self):
        B = np.array([0.1, 0.05], dtype=np.float32)

        X = xlayer.XLayer(
            name='bias_add',
            type=['BiasAdd'],
            shapes=[1, 2, 1, 1],
            sizes=[2],
            bottoms=['input'],
            tops=[],
            data=[B],
            attrs={
                'axis': 1
            },
            targets=[]
        )

        input_shapes = {
            'input': TensorShape([1, 2, 1, 1])
        }
        inputs = {
            'input': np.array([1, 1], dtype=np.float32).reshape(1, 2, 1, 1)
        }
        params = {
            'bias_add_bias': B
        }
        layers = base.get_bias_add_layer(BiasAddLayer,
                                         ConstantLayer)(
            X, input_shapes, params)
        assert(len(layers) == 2)

        inputs.update(params)
        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

        expected_outpt = np.array([1.1, 1.05], dtype=np.float32)\
            .reshape(1, 2, 1, 1)

        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_dense_layer(self):

        iX = px.ops.input('input', shape=[1, 4])
        wX = px.ops.constant('W', np.array([[1, 0, 3, -4], [8, -1, -1, -1]],
                                           dtype=np.float32))
        X = px.ops.dense('test_dense', iX, wX, units=2)

        input_shapes = {
            'input': TensorShape([1, 4])
        }
        inputs = {
            'input': np.array([1, 0, -1, 4], dtype=np.float32).reshape(1, 4)
        }
        params = {
            'test_dense_weights': np.array([[1, 0, 3, -4], [8, -1, -1, -1]],
                                           dtype=np.float32),
            'test_dense_biases': np.array([1, -1], dtype=np.float32)
        }
        layers = base.get_dense_layer(DenseLayer,
                                      ConstantLayer)(
            X, input_shapes, params)
        assert(len(layers) == 3)

        inputs.update(params)
        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

        expected_outpt = np.expand_dims(np.array([-17, 4]), 0)

        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_dense_layer_expand_squeeze(self):

        iX = px.ops.input('input', shape=[1, 4])
        wX = px.ops.constant('W', np.array([[1, 0, 3, -4], [8, -1, -1, -1]],
                                           dtype=np.float32))
        X = px.ops.dense('test_dense', iX, wX, units=2)

        input_shapes = {
            'input': TensorShape([1, 4])
        }
        inputs = {
            'input': np.array([[1, 0, -1, 4]], dtype=np.float32)
        }
        params = {
            'test_dense_weights': np.array([[1, 0, 3, -4], [8, -1, -1, -1]],
                                           dtype=np.float32),
            'test_dense_biases': np.array([1, -1], dtype=np.float32)
        }
        layers = base.get_dense_layer(DenseLayer,
                                      ConstantLayer)(
            X, input_shapes, params)
        assert(len(layers) == 3)

        inputs.update(params)
        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

        expected_outpt = np.array([[-17, 4]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_expand_dims(self):

        X = xlayer.XLayer(
            type=['ExpandDims'],
            name='ed1',
            shapes=[-1, 1, 1, 4],
            sizes=[4],
            bottoms=['in1'],
            tops=[],
            targets=[],
            attrs={'axis': 1, 'num_newaxis': 2}
        )

        input_shapes = {'in1': TensorShape([1, 4])}

        layers = X_2_TF['ExpandDims'](X, input_shapes, {})
        assert(len(layers) == 1)

        inputs = [np.reshape(np.array([[1, 1], [5, -1]], dtype=np.float32),
                             (1, 4))]

        outpt = layers[0].forward_exec(inputs)

        assert (outpt.shape) == (1, 1, 1, 4)

    def test_relu(self):
        iX = px.ops.input('input', shape=[1, 1, 4, 4])
        X = px.ops.relu('test_relu', [iX])
        input_shapes = {'input': TensorShape([1, 1, 4, 4])}
        inputs = [np.reshape(np.array([[1, 1, 0, -4], [5, 1, 0, -8],
                                       [3, -5, 1, 0], [1, 9, 3, 4]],
                                      dtype=np.float32),
                             (1, 1, 4, 4))]

        layers = X_2_TF['ReLU'](X, input_shapes, {})

        assert len(layers) == 1
        outpt = layers[0].forward_exec(inputs)

        expected_outpt = np.reshape(np.array([[1, 1, 0, 0.], [5, 1, 0, 0.],
                                              [3, 0., 1, 0], [1, 9, 3, 4]],
                                             dtype=np.float32),
                                    (1, 1, 4, 4))
        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_expand_dims(self):

        X = xlayer.XLayer(
            type=['Sigmoid'],
            name='s1',
            shapes=[-1, 1, 2, 2],
            sizes=[4],
            bottoms=['in1'],
            tops=[],
            targets=[],
            attrs={}
        )

        input_shapes = {'in1': TensorShape([1, 1, 2, 2])}

        layers = X_2_TF['Sigmoid'](X, input_shapes, {})
        assert len(layers) == 1

        inpt = np.array([[1, 10], [5, -1]], dtype=np.float32)\
            .reshape(1, 1, 2, 2)
        expected_outpt = 1 / (1 + np.exp(-inpt))

        outpt = layers[0].forward_exec([inpt])

        assert (outpt.shape) == (1, 1, 2, 2)
        np.testing.assert_array_almost_equal(outpt, expected_outpt)
