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

import os
import unittest
import numpy as np

try:
    import tensorflow as tf
    from pyxir.shapes import TensorShape, TupleShape
    from pyxir.runtime.tensorflow.rt_layer_tf import *
    from pyxir.runtime.tensorflow.x_2_tf_registry import *
    from pyxir.runtime.tensorflow.ops.tf_l0_input_and_other import *
    from pyxir.runtime.tensorflow.ops.tf_l1_basic_nn import *
    from pyxir.runtime.tensorflow.ops.tf_l2_convolutions import *
    from pyxir.runtime.tensorflow.ops.tf_l3_math_and_transform import *
    from pyxir.runtime.tensorflow.ops.tf_l4_broadcast_and_reductions import *
    from pyxir.runtime.tensorflow.ops.tf_l5_vision import *
    from pyxir.runtime.tensorflow.ops.tf_l11_quant import *
    skip_tf = False
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Tensorflow related test because Tensorflow is not available")
    skip_tf = True


try:
    from pyxir.io.cvx import ImgLoader
    skip_cvx = False
except ModuleNotFoundError:
    skip_cvx = True

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def softmax(x):
    x_exp = np.exp(x - np.max(x))

    # return np.exp(x) / np.sum(np.exp(x), axis=0)
    return x_exp / x_exp.sum()


class TestRtLayerTF(unittest.TestCase):

    
    def test_batch_norm(self):
        M = np.array([0.5, 1.2], dtype=np.float32)
        V = np.array([0.1, 0.05], dtype=np.float32)
        G = np.array([2.0, 1.0], dtype=np.float32)
        B = np.array([1., -1.0], dtype=np.float32)

        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 1, 1])],
                subgraph=None
            ),
            ConstantLayer(
                name='mean',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=M
            ),
            ConstantLayer(
                name='var',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=V
            ),
            ConstantLayer(
                name='gamma',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=G
            ),
            ConstantLayer(
                name='beta',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=B
            ),
            BatchNormLayer(
                name='bn',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input', 'mean', 'var', 'gamma', 'beta'],
                input_shapes=[TensorShape([1, 2, 1, 1]),
                              TensorShape([2]),
                              TensorShape([2]),
                              TensorShape([2]),
                              TensorShape([2])],
                subgraph=None,
                attrs={
                    'axis': 1
                },
                mean=None,
                variance=None,
                gamma=None,
                beta=None,
                variance_epsilon=0.0000001
            )
        ]

        inputs = {
            'input': np.ones((1, 2, 1, 1), dtype=np.float32)
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        expected_outpt = np.reshape(G, (1, 2, 1, 1)) *\
            (np.ones((1, 2, 1, 1), dtype=np.float32) -
             np.reshape(M, (1, 2, 1, 1))) /\
            np.reshape(np.sqrt(V + 0.0000001), (1, 2, 1, 1)) +\
            np.reshape(B, (1, 2, 1, 1))

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_bias_add(self):
        B = np.array([0.1, 0.05], dtype=np.float32)

        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 1, 1])],
                subgraph=None
            ),
            ConstantLayer(
                name='bias',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=B
            ),
            BiasAddLayer(
                name='bias_add',
                xtype='BiasAdd',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input', 'bias'],
                input_shapes=[TensorShape([1, 2, 1, 1]),
                                         TensorShape([2])],
                data=[],
                subgraph=None,
                attrs={
                    'axis': 1
                }
            )
        ]

        inputs = {
            'input': np.ones((1, 2, 1, 1), dtype=np.float32)
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        expected_outpt = np.reshape(np.array([1.1, 1.05], dtype=np.float32),
                                    (1, 2, 1, 1))

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_concat_layer(self):
        layers = [
            InputLayer(
                name='input1',
                shape=TensorShape([1, 1, 2, 2]),
                dtype='float32',
                inputs=['input1'],
                input_shapes=[TensorShape([1, 1, 2, 2])],
                subgraph=None
            ),
            InputLayer(
                name='input2',
                shape=TensorShape([1, 1, 2, 2]),
                dtype='float32',
                inputs=['input2'],
                input_shapes=[TensorShape([1, 1, 2, 2])],
                subgraph=None
            ),
            ConcatLayer(
                name='concat1',
                xtype='Concat',
                shape=TensorShape([1, 2, 2, 2]),
                dtype='float32',
                inputs=['input1', 'input2'],
                input_shapes=[TensorShape([1, 1, 2, 2]),
                                         TensorShape([1, 1, 2, 2])],
                data=[],
                subgraph=None,
                attrs={'axis': 1}
            )
        ]

        inputs = [np.array([[[[1, 2], [3, 4]]]]),
                  np.array([[[[5, 6], [7, 8]]]])]
        inpt1 = layers[0].forward_exec([inputs[0]])
        inpt2 = layers[1].forward_exec([inputs[1]])
        outpt = layers[2].forward_exec([inpt1, inpt2])

        expected_outpt = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_conv2d_nchw(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            ConvLayer(
                name='conv1',
                shape=TensorShape([2, 1, 3, 3]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None,
                attrs={
                    'data_layout': 'NCHW'
                },
                kernel=np.reshape(
                    np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             dtype=np.float32),
                    (2, 1, 2, 2)),
                kernel_layout='OIHW',
                kernel_groups=1,
                biases=np.array([0, 0], dtype=np.float32),
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
                strides=[1, 1, 1, 1],
                dilations=[1, 1, 1, 1]
            )
        ]

        inputs = [np.ones((1, 1, 4, 4), dtype=np.float32)]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[10., 10., 10.],
                                     [10., 10., 10.],
                                     [10., 10., 10.]],
                                    [[26., 26., 26.],
                                     [26., 26., 26.],
                                     [26., 26., 26.]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_conv2d_nhwc(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 4, 4, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 4, 4, 1])],
                subgraph=None
            ),
            ConvLayer(
                name='conv1',
                shape=TensorShape([1, 3, 3, 2]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 4, 4, 1])],
                subgraph=None,
                attrs={
                    'data_layout': 'NHWC'
                },
                kernel=np.reshape(
                    np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                             dtype=np.float32),
                    (2, 1, 2, 2)),
                kernel_layout='OIHW',
                kernel_groups=1,
                biases=np.array([0, 0], dtype=np.float32),
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
                strides=[1, 1, 1, 1],
                dilations=[1, 1, 1, 1]
            )
        ]

        inputs = [np.ones((1, 4, 4, 1), dtype=np.float32)]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.transpose(
            np.array([[[[10., 10., 10.],
                        [10., 10., 10.],
                        [10., 10., 10.]],
                       [[26., 26., 26.],
                        [26., 26., 26.],
                        [26., 26., 26.]]]]),
            (0, 2, 3, 1))

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_depthwise_conv2d_nchw(self):
        W = np.reshape(
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
            (2, 1, 2, 2)
        )
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 2, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 4, 4])],
                subgraph=None
            ),
            ConvLayer(
                name='conv1',
                shape=TensorShape([1, 2, 3, 3]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 4, 4])],
                subgraph=None,
                attrs={
                    'data_layout': 'NCHW'
                },
                kernel=W,
                kernel_layout='OIHW',
                kernel_groups=2,
                biases=np.array([0, 0], dtype=np.float32),
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
                strides=[1, 1, 1, 1],
                dilations=[1, 1, 1, 1]
            )
        ]

        inputs = [np.ones((1, 2, 4, 4), dtype=np.float32)]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[10., 10., 10.],
                                     [10., 10., 10.],
                                     [10., 10., 10.]],
                                    [[26., 26., 26.],
                                     [26., 26., 26.],
                                     [26., 26., 26.]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_depthwise_conv2d_nhwc(self):
        W = np.reshape(
            np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
            (2, 1, 2, 2)
        )
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 4, 4, 2]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 4, 4, 2])],
                subgraph=None
            ),
            ConvLayer(
                name='conv1',
                shape=TensorShape([1, 3, 3, 2]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 4, 4, 2])],
                subgraph=None,
                attrs={
                    'data_layout': 'NHWC'
                },
                kernel=W,
                kernel_layout='OIHW',
                kernel_groups=2,
                biases=np.array([0, 0], dtype=np.float32),
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
                strides=[1, 1, 1, 1],
                dilations=[1, 1, 1, 1]
            )
        ]

        inputs = [np.ones((1, 4, 4, 2), dtype=np.float32)]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.transpose(
            np.array([[[[10., 10., 10.],
                        [10., 10., 10.],
                        [10., 10., 10.]],
                       [[26., 26., 26.],
                        [26., 26., 26.],
                        [26., 26., 26.]]]]),
            (0, 2, 3, 1))

        np.testing.assert_array_equal(outpt, expected_outpt)

    @unittest.skipIf(skip_cvx or skip_tf, "Skipping Cvx related test because cvx or tensorflow is"
                    "not available")
    def test_cvx_layer_nchw(self):
        layers = [
            CvxLayer(
                name='cvx',
                xtype='Cvx',
                shape=TensorShape([-1, 3, 225, 225]),
                dtype='float32',
                inputs=['cvx'],
                input_shapes=[],
                data=[],
                subgraph=None,
                attrs={
                    'cvx_key': 'scale-0.5__transpose-2,0,1'
                }
            )
        ]
        assert(layers[0].res.get_shape().as_list() == [None, 3, 225, 225])

        test_img = os.path.join(FILE_PATH, '../../../images/v.png')

        inputs = {
            'cvx': [test_img]
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        assert(outpt.shape == (1, 3, 225, 225))

    @unittest.skipIf(skip_cvx or skip_tf, "Skipping Cvx related test because cvx or tensorflow is"
                    "not available")
    def test_cvx_layer_nhwc(self):
        layers = [
            CvxLayer(
                name='cvx',
                xtype='Cvx',
                shape=TensorShape([-1, 225, 225, 3]),
                dtype='float32',
                inputs=['cvx'],
                input_shapes=[],
                data=[],
                subgraph=None,
                attrs={
                    'cvx_key': 'scale-0.5',
                    'data_layout': 'NHWC'
                }
            )
        ]
        assert(layers[0].res.get_shape().as_list() == [None, 225, 225, 3])

        test_img = os.path.join(FILE_PATH, '../../../images/v.png')

        inputs = {
            'cvx': [test_img]
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        assert(outpt.shape == (1, 225, 225, 3))

    
    def test_dense_layer(self):

        W = np.array([[1., 3., 0., -7.], [2., -4., 6., 8.]], dtype=np.float32)
        B = np.array([-1., -1.], dtype=np.float32)

        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 4])],
                subgraph=None
            ),
            ConstantLayer(
                name='dense1_weights',
                shape=TensorShape([2, 4]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=W
            ),
            ConstantLayer(
                name='dense1_biases',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=B
            ),
            DenseLayer(
                name='dense1',
                shape=TensorShape([1, 2]),
                dtype='float32',
                inputs=['input', 'dense1_weights', 'dense1_biases'],
                input_shapes=[TensorShape([1, 4]),
                                         TensorShape([2, 4]),
                                         TensorShape([2])],
                subgraph=None,
                data_layout='NC',
                weights=W,
                kernel_layout='OI',
                biases=B,
                use_relu=False
            ),
            OutputLayer(
                name='output',
                xtype='Output',
                shape=TensorShape([1, 2]),
                dtype='float32',
                inputs=['dense1'],
                input_shapes=[TensorShape([1, 2])],
                data=[],
                subgraph=None,
                attrs={}
            ),
        ]

        inputs = {
            'input': np.ones((1, 4), dtype=np.float32)
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        expected_outpt = np.array([[-4.0, 11.]], dtype=np.float32)

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_leaky_relu(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            LeakyReluLayer(
                name='leaky_relu',
                xtype='ReLU',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={
                    'alpha': 0.1
                }
            )
        ]

        inputs = [np.reshape(
                        np.array([1, -1, 0, 4, -5, 1, 0, 8, 3,
                                  -5, 1, 0, 1, 9, -3, -4],
                                 dtype=np.float32),
                        (1, 1, 4, 4))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1, -0.1, 0, 4], [-0.5, 1, 0, 8],
                                     [3, -0.5, 1, 0], [1, 9, -0.3, -0.4]]]])

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_maxpool_layer(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            PoolingLayer(
                name='conv1',
                shape=TensorShape([1, 1, 3, 3]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None,
                op='Max',
                ksize=[1, 1, 2, 2],
                paddings=[[0, 0], [0, 0], [1, 1], [1, 1]],
                strides=[1, 1, 2, 2],
                attrs={
                    'data_layout': 'NCHW'
                }
            )
        ]

        inputs = [np.reshape(np.array([[1, 1, 0, 4], [5, 1, 0, 8],
                                       [3, 5, 1, 0], [1, 9, 3, 4]],
                                      dtype=np.float32),
                             (1, 1, 4, 4))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1, 1, 4], [5, 5, 8], [1, 9, 4]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_mean(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            MeanLayer(
                name='mean',
                xtype='Mean',
                shape=TensorShape([1, 1, 1, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={
                    'axes': [0, 1, 2, 3],
                    'keepdims': True,
                    'exclude': False
                }
            )
        ]

        data = np.reshape(np.array([[1, -1, 0, 4, -5, 1, 0, 8, 3,
                                        -5, 1, 0, 1, 9, -3, -4]],
                                   dtype=np.float32),
                          (1, 1, 4, 4))
        inputs = [data]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.mean(data, axis=(0, 1, 2, 3), keepdims=True)

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_pad(self):
        padding = ((0, 0), (0, 0), (0, 1), (0, 1))

        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 1, 1])],
                subgraph=None
            ),
            PadLayer(
                name='pad',
                xtype='Pad',
                shape=TensorShape([1, 2, 3, 3]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 1, 1])],
                data=[],
                subgraph=None,
                attrs={
                    'padding': padding
                }
            )
        ]

        inputs = {
            'input': np.ones((1, 2, 1, 1), dtype=np.float32)
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        expected_outpt = np.pad(inputs['input'], padding, mode='constant')

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_pool_no_division_layer(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            PoolingNoDivisionLayer(
                name='pool1',
                shape=TensorShape([1, 1, 3, 3]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None,
                op='Avg',
                ksize=[1, 1, 2, 2],
                paddings=[[0, 0], [0, 0], [1, 1], [1, 1]],
                strides=[1, 1, 2, 2],
                attrs={
                    'data_layout': 'NCHW',
                    'op': 'Avg'
                }
            )
        ]

        inputs = [np.reshape(np.array([[1, 1, 0, 4], [5, 1, 0, 8],
                                       [3, 5, 1, 0], [1, 9, 3, 4]],
                                      dtype=np.float32),
                             (1, 1, 4, 4))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1, 1, 4], [8, 7, 8], [1, 12, 4]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_relu(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            ReluLayer(
                name='relu1',
                xtype='ReLU',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={}
            )
        ]

        inputs = [np.reshape(
                        np.array([1, -1, 0, 4, -5, 1, 0, 8, 3,
                                  -5, 1, 0, 1, 9, -3, -4],
                                 dtype=np.float32),
                        (1, 1, 4, 4))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1, 0, 0, 4], [0, 1, 0, 8],
                                     [3, 0, 1, 0], [1, 9, 0, 0]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_relu6(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            Relu6Layer(
                name='relu1',
                xtype='ReLU6',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={}
            )
        ]

        inputs = [np.reshape(
                        np.array([1, -1, 0, 4, -5, 1, 0, 8, 3,
                                  -5, 1, 0, 1, 9, -3, -4],
                                 dtype=np.float32),
                        (1, 1, 4, 4))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1, 0, 0, 4], [0, 1, 0, 6],
                                     [3, 0, 1, 0], [1, 6, 0, 0]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_scale(self):
        G = np.array([0.5, 1.2], dtype=np.float32)
        B = np.array([0.1, 0.05], dtype=np.float32)

        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 2, 1, 1])],
                subgraph=None
            ),
            ConstantLayer(
                name='gamma',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=G
            ),
            ConstantLayer(
                name='beta',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                subgraph=None,
                value=B
            ),
            ScaleLayer(
                name='scale',
                shape=TensorShape([1, 2, 1, 1]),
                dtype='float32',
                inputs=['input', 'gamma', 'beta'],
                input_shapes=[TensorShape([1, 2, 1, 1]),
                                         TensorShape([2]),
                                         TensorShape([2])],
                subgraph=None,
                attrs={
                    'axis': 1
                },
                gamma=None,
                beta=None
            )
        ]

        inputs = {
            'input': np.ones((1, 2, 1, 1), dtype=np.float32)
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        expected_outpt = (np.ones((1, 2, 1, 1), dtype=np.float32) *
                          np.reshape(G, (1, 2, 1, 1))) +\
            np.reshape(B, (1, 2, 1, 1))

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_softmax_layer(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([16]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([16])],
                subgraph=None
            ),
            SoftmaxLayer(
                name='softmax1',
                xtype='Softmax',
                shape=TensorShape([16]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([16])],
                data=[],
                subgraph=None,
                attrs={}
            )
        ]

        inpt = np.array([1, -1, 0, 4, -5, 1, 0, 8, 3, -5, 1, 0, 1, 9, -3, -4],
                        dtype=np.float32)

        inputs = [inpt]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = softmax(inpt)

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_squeeze(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 1, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 1, 4])],
                subgraph=None
            ),
            SqueezeLayer(
                name='squeeze',
                xtype='Squeeze',
                shape=TensorShape([1, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 1, 4])],
                data=[],
                subgraph=None,
                attrs={
                    'axis': [1, 2]
                }
            )
        ]

        inputs = [np.reshape(np.array([1, -1, 0, 4], dtype=np.float32),
                             (1, 1, 1, 4))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.reshape(np.array([1, -1, 0, 4], dtype=np.float32),
                                    (1, 4))

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_tanh(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            TanhLayer(
                name='tanh1',
                xtype='Tanh',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={}
            )
        ]

        a = np.reshape(
                np.array([1, -1, 0, 4, -5, 1, 0, 8, 3,
                          -5, 1, 0, 1, 9, -3, -4],
                         dtype=np.float32),
                (1, 1, 4, 4))
        inputs = [a]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.tanh(a)

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_transpose(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            TransposeLayer(
                name='transpose',
                xtype='Transpose',
                shape=TensorShape([1, 4, 4, 1]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={
                    'axes': [0, 2, 3, 1]
                }
            )
        ]

        data = np.reshape(np.array([[1, -1, 0, 4, -5, 1, 0, 8, 3,
                                        -5, 1, 0, 1, 9, -3, -4]],
                                   dtype=np.float32),
                          (1, 1, 4, 4))
        inputs = [data]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.transpose(data, (0, 2, 3, 1))

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_conv2d_transpose(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 3, 3]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 3, 3])],
                subgraph=None
            ),
            Conv2DTransposeLayer(
                name='trans_conv',
                shape=TensorShape([1, 1, 6, 6]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 3, 3])],
                subgraph=None,
                attrs={
                    'data_layout': 'NCHW'
                },
                kernel=np.reshape(
                    np.array([[[1, 1, 1], [1, 2, 1], [1, 1, 1]]],
                             dtype=np.float32),
                    (1, 1, 3, 3)),
                kernel_layout='OIHW',
                kernel_groups=1,
                biases=np.array([0], dtype=np.float32),
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
                strides=[1, 1, 2, 2],
                dilations=[1, 1, 1, 1]
            )
        ]

        data = np.reshape(np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]], dtype=np.float32),
                          (1, 1, 3, 3))
        # NOTE: VALID
        # data1 = np.reshape(np.array([[1, 2], [3, 4]], dtype=np.float32),
        #                    (1, 1, 2, 2))

        inputs = [data]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1, 1, 3, 2, 5, 3],
                                     [1, 2, 3, 4, 5, 6],
                                     [5, 5, 12, 7, 16, 9],
                                     [4, 8, 9, 10, 11, 12],
                                     [11, 11, 24, 13, 28, 15],
                                     [7, 14, 15, 16, 17, 18]]]],
                                  dtype=np.float32)

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_tuple(self):
        layers = [
            InputLayer(
                name='in1',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['in1'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            InputLayer(
                name='in2',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['in2'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            TupleLayer(
                name='tuple',
                xtype='Tuple',
                shape=TupleShape([TensorShape([1, 1, 4, 4]),
                                  TensorShape([1, 1, 4, 4])]),
                dtype='float32',
                inputs=['in1', 'in2'],
                input_shapes=[TensorShape([1, 1, 4, 4]),
                                         TensorShape([1, 1, 4, 4])],
                data=[],
                subgraph=None,
                attrs={}
            ),
        ]

        in1 = np.reshape(np.array([[1, -1, 0, 4, -5, 1, 0, 8, 3,
                                    -5, 1, 0, 1, 9, -3, -4]],
                                  dtype=np.float32),
                         (1, 1, 4, 4))
        in2 = np.reshape(np.array([[1, 0, 0, 4, 0, 1, 0, 8, 3,
                                    0, 1, 0, 1, 9, 0, 0]],
                                  dtype=np.float32),
                         (1, 1, 4, 4))

        inputs = {
            'in1': in1,
            'in2': in2
        }

        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            inputs[layer.name] = outpt

        np.testing.assert_array_equal(inputs['tuple'][0], in1)
        np.testing.assert_array_equal(inputs['tuple'][1], in2)

    
    def test_variable(self):
        tf.compat.v1.reset_default_graph()
        data = np.reshape(np.array([[1, -1, 0, 4, -5, 1, 0, 8, 3,
                                        -5, 1, 0, 1, 9, -3, -4]],
                                   dtype=np.float32),
                          (1, 1, 4, 4))

        layers = [
            VariableLayer(
                name='var',
                xtype='Variable',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=[],
                input_shapes=[],
                data=[data],
                subgraph=None,
                attrs={}
            )
        ]

        outpt = layers[0].forward_exec([])

        np.testing.assert_array_equal(outpt, data)

    # Quantization layer
    
    def test_quantize_layer(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 2, 2]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 2, 2])],
                subgraph=None
            ),
            QuantizeLayer(
                name='quant1',
                shape=TensorShape([1, 1, 2, 2]),
                dtype='int8',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 2, 2])],
                subgraph=None,
                input_types=['float32'],
                threshold=[2.0],
                axis=1,
                bitwidth=8
            )
        ]

        inputs = [np.reshape(np.array([[1.5, 4], [-1, 2]],
                                      dtype=np.float32),
                             (1, 1, 2, 2))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[95, 127], [-63, 127]]]])

        np.testing.assert_array_equal(outpt, expected_outpt)

    
    def test_unquantize_layer(self):
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 2, 2]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 2, 2])],
                subgraph=None
            ),
            UnQuantizeLayer(
                name='unquant1',
                shape=TensorShape([1, 1, 2, 2]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 2, 2])],
                subgraph=None,
                input_types=['int8'],
                threshold=[2.0],
                axis=1,
                bitwidth=8
            )
        ]

        inputs = [np.reshape(np.array([[95, 127], [-63, 127]],
                                      dtype=np.int8),
                             (1, 1, 2, 2))]
        for layer in layers:
            outpt = layer.forward_exec(inputs)
            inputs = [outpt]

        expected_outpt = np.array([[[[1.4960629, 2.0], [-0.9921259, 2.0]]]],
                                  dtype=np.float32)

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    
    def test_quantized_conv_layer(self):
        quant_params = {
            "bw_layer_in": 8,
            "bw_layer_out": 8,
            "bw_params": 8,
            "name": "conv2d0",
            "postscale_shift": [
                22,
                24
            ],
            "prescale_shift": [
                0,
                0
            ],
            "scale": [
                16584,
                22112
            ],
            "sf_layer_in": 1.220472440944882,
            "sf_layer_out": 7.291338582677166,
            "sf_params": [
                0.023622047156095505,
                0.007874015718698502
            ],
            "th_layer_in": 155.0,
            "th_layer_out": 926.0,
            "th_params": [
                3.0,
                1.0
            ]
        }
        layers = [
            InputLayer(
                name='input',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='float32',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None
            ),
            QuantizeLayer(
                name='input_quant',
                shape=TensorShape([1, 1, 4, 4]),
                dtype='int8',
                inputs=['input'],
                input_shapes=[TensorShape([1, 1, 4, 4])],
                subgraph=None,
                input_types=['float32'],
                threshold=[quant_params['th_layer_in']],
                axis=1,
                bitwidth=8
            ),
            InputLayer(
                name='kernel',
                shape=TensorShape([2, 1, 2, 2]),
                dtype='float32',
                inputs=['kernel'],
                input_shapes=[TensorShape([2, 1, 2, 2])],
                subgraph=None
            ),
            QuantizeLayer(
                name='kernel_quant',
                shape=TensorShape([2, 1, 2, 2]),
                dtype='int8',
                inputs=['kernel'],
                input_shapes=[TensorShape([2, 1, 2, 2])],
                subgraph=None,
                input_types=['float32'],
                threshold=quant_params['th_params'],
                axis=0,
                bitwidth=8
            ),
            InputLayer(
                name='bias',
                shape=TensorShape([2]),
                dtype='float32',
                inputs=['bias'],
                input_shapes=[TensorShape([2])],
                subgraph=None
            ),
            QuantizeBiasLayer(
                name='bias_quant',
                shape=TensorShape([2]),
                dtype='int32',
                inputs=['bias'],
                input_shapes=[TensorShape([2])],
                subgraph=None,
                input_types=['float32'],
                threshold_bias=quant_params['th_params'],
                threshold_ext=quant_params['th_layer_in'],
                bitwidth=8,
                do_rounding=True
            ),
            ConvLayer(
                name='conv1',
                shape=TensorShape([1, 2, 3, 3]),
                dtype='float32',
                inputs=['input_quant', 'kernel_quant', 'bias_quant'],
                input_shapes=[TensorShape([1, 1, 4, 4]),
                                         TensorShape([2, 1, 2, 2]),
                                         TensorShape([2])],
                subgraph=None,
                attrs={'data_layout': 'NCHW'},
                kernel=None,
                kernel_layout='OIHW',
                kernel_groups=1,
                biases=None,
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0]],
                strides=[1, 1, 1, 1],
                dilations=[1, 1, 1, 1]
            ),
            QuantizeInterLayer(
                name='conv1_quant',
                shape=TensorShape([1, 2, 3, 3]),
                dtype='int8',
                inputs=['conv1'],
                input_shapes=[TensorShape([1, 2, 3, 3])],
                subgraph=None,
                prescale_shift=quant_params['prescale_shift'],
                scale=quant_params['scale'],
                postscale_shift=quant_params['postscale_shift'],
                axis=1,
                bitwidth=8
            ),
            UnQuantizeLayer(
                name='output',
                shape=TensorShape([1, 2, 3, 3]),
                dtype='float32',
                inputs=['conv1_quant'],
                input_shapes=[TensorShape([1, 2, 3, 3])],
                subgraph=None,
                input_types=['int8'],
                threshold=[quant_params['th_layer_out']],
                axis=0,
                bitwidth=8
            )
        ]

        inputs = {
            'input': np.reshape(np.array([
                        [10, 10, 0, 40],
                        [50, 10, 0, 80],
                        [30, 50, 10, 0],
                        [10, 90, 30, 40]]), (1, 1, 4, 4)),
            'kernel': np.reshape(
                np.array([[[1, 2], [3, 0]], [[1, 1], [0, 1]]],
                         dtype=np.float32),
                (2, 1, 2, 2)),
            'bias': np.array([0., 0.])
        }

        for layer in layers:

            # print("-----------------------")
            # print("Run layer: {}".format(layer.name))

            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

            # print("Output:", outpt.shape, outpt)

            inputs[layer.name] = outpt

        expected_outpt = np.array([[
            [[174.99213, 36.45669, 80.20473],
             [153.1181, 153.1181, 189.57481],
             [153.1181, 335.40158,  94.78741]],

            [[29.165354, 7.2913384, 116.661415],
             [109.37008, 21.874016, 80.20473],
             [167.70079, 87.49606, 51.03937]]]],
            dtype=np.float32)

        np.testing.assert_array_almost_equal(outpt, expected_outpt, decimal=4)

if __name__ == '__main__':
    unittest.main()
