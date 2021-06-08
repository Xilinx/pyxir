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
    import tensorflow as tf
    from pyxir.runtime.tensorflow.x_2_tf_registry import *
    from pyxir.runtime.tensorflow.ops.tf_l0_input_and_other import *
    from pyxir.runtime.tensorflow.ops.tf_l2_convolutions import *
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Tensorflow related test because Tensorflow is not available")


class TestTfL2Convolutions(unittest.TestCase):

    def create_conv2d_layers(self, in_name, in_shape, weights, strides, padding, dilation,
                             groups, data_layout="NCHW", kernel_layout="OIHW",
                             target_kernel_layout="OIHW"):
        tf.compat.v1.reset_default_graph()
        in_layer = px.ops.input("in", shape=in_shape[:])
        weights_layer = px.ops.constant("w", weights)
        kernel_size = [weights_layer.shapes[kernel_layout.index("H")],
                       weights_layer.shapes[kernel_layout.index("W")]]
        channels = weights_layer.shapes[kernel_layout.index("O")]
        X = px.ops.conv2d("conv", in_layer, weights_layer, kernel_size, strides,
                          padding, dilation, groups, channels, data_layout, kernel_layout,
                          target_kernel_layout)
        
        input_shapes = {in_name: TensorShape(in_shape)}
        params = {
            'conv_kernel': weights,
            'conv_biases': np.array([0, 0], dtype=np.float32)
        }
        layers = base.get_conv2d_layer(ConvLayer, ConstantLayer)(X, input_shapes, params)
        return layers

    def execute_layers(self, layers, inputs):
        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            out = layer.forward_exec(inpts)
            inputs[layer.name] = out
        return out

    def test_conv2d_basic(self):
        K = np.reshape(np.array([[[1, 2], [3, 4]],
                                 [[5, 6], [7, 8]]],
                                dtype=np.float32),
                       (2, 1, 2, 2))
        
        in_shape = [1, 1, 4, 4]
        layers = self.create_conv2d_layers(
            in_name="in",
            in_shape=in_shape,
            weights=K,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1
        )
        assert len(layers) == 3
        inputs = {"in": np.ones(in_shape, dtype=np.float32)}
        
        out = self.execute_layers(layers, inputs)
        expected_outpt = np.array([[[[10., 10., 10.],
                                     [10., 10., 10.],
                                     [10., 10., 10.]],
                                    [[26., 26., 26.],
                                     [26., 26., 26.],
                                     [26., 26., 26.]]]])

        np.testing.assert_array_equal(out, expected_outpt)

    def test_conv2d_dilation(self):
        K = np.reshape(np.array([[[1, 2], [3, 4]],
                                 [[5, 6], [7, 8]]],
                                dtype=np.float32),
                       (2, 1, 2, 2))
        
        in_shape = [1, 1, 4, 4]
        layers = self.create_conv2d_layers(
            in_name="in",
            in_shape=in_shape,
            weights=K,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[2, 2],
            groups=1
        )
        assert len(layers) == 3
        inputs = {"in": np.ones(in_shape, dtype=np.float32)}
        
        out = self.execute_layers(layers, inputs)
        expected_outpt = np.array([[[[10., 10.],
                                     [10., 10.]],
                                    [[26., 26.],
                                     [26., 26.]]]])

        np.testing.assert_array_equal(out, expected_outpt)

    def test_conv2d_tfl(self):
        tf.compat.v1.reset_default_graph()
        K = np.transpose(np.reshape(np.array([[[1, 2], [3, 4]],
                                              [[5, 6], [7, 8]]],
                                             dtype=np.float32),
                                    (2, 1, 2, 2)),
                         (0, 2, 3, 1))

        in_shape = [1, 4, 4, 1]
        layers = self.create_conv2d_layers(
            in_name="in",
            in_shape=in_shape,
            weights=K,
            strides=[1, 1],
            padding=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            data_layout="NHWC",
            kernel_layout="OHWI",
            target_kernel_layout="OHWI"
        )
        assert len(layers) == 3
        inputs = {"in": np.ones(in_shape, dtype=np.float32)}
        
        out = self.execute_layers(layers, inputs)

        expected_outpt = np.transpose(np.array([[[[10., 10., 10.],
                                                  [10., 10., 10.],
                                                  [10., 10., 10.]],
                                                 [[26., 26., 26.],
                                                  [26., 26., 26.],
                                                  [26., 26., 26.]]]]),
                                      (0, 2, 3, 1))

        np.testing.assert_array_equal(out, expected_outpt)

    def test_conv2d_transpose(self):
        tf.compat.v1.reset_default_graph()
        K = np.reshape(np.array([[[1, 1, 1], [1, 2, 1], [1, 1, 1]]],
                                dtype=np.float32),
                       (1, 1, 3, 3))
        B = np.array([1], dtype=np.float32)

        X = xlayer.XLayer(
            name='tconv',
            type=['TransposeConv2D'],
            shapes=[1, 1, 6, 6],
            sizes=[25],
            bottoms=['input'],
            tops=[],
            data=xlayer.ConvData(K, B),
            attrs={
                'data_layout': 'NCHW',
                'kernel_layout': 'OIHW',
                'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                'strides': [2, 2],
                'dilation': [1, 1],
                'groups': 1
            },
            targets=[]
        )

        input_shapes = {
            'input': TensorShape([1, 1, 3, 3])
        }
        data = np.reshape(np.array([[1, 2, 3],
                                    [4, 5, 6],
                                    [7, 8, 9]], dtype=np.float32),
                          (1, 1, 3, 3))
        inputs = {
            'input': data
        }
        params = {
            'tconv_kernel': K,
            'tconv_biases': B
        }
        layers = base.get_conv2d_transpose_layer(
            Conv2DTransposeLayer, ConstantLayer)(X, input_shapes, params)
        assert(len(layers) == 3)

        inputs.update(params)
        for layer in layers:
            inpts = [inputs[name] for name in layer.inputs]
            outpt = layer.forward_exec(inpts)

        expected_outpt = np.array([1], dtype=np.float32) +\
            np.array([[[[1, 1, 3, 2, 5, 3],
                        [1, 2, 3, 4, 5, 6],
                        [5, 5, 12, 7, 16, 9],
                        [4, 8, 9, 10, 11, 12],
                        [11, 11, 24, 13, 28, 15],
                        [7, 14, 15, 16, 17, 18]]]],
                     dtype=np.float32)

        np.testing.assert_array_almost_equal(outpt, expected_outpt)

    def test_maxpool_stride_1(self):

        iX = px.ops.input('input', shape=[1, 1, 4, 4])

        X = px.ops.pool2d(
            op_name='test_maxpool',
            input_layer=iX,
            pool_type='Max',
            pool_size=[2, 2],
            strides=[1, 1],
            padding=[0, 0],
            layout='NCHW'
        )

        input_shapes = {
            'input': TensorShape([1, 1, 4, 4])
        }
        inputs = [np.reshape(np.array([[1, 1, 0, 4], [5, 1, 0, 8],
                                       [3, 5, 1, 0], [1, 9, 3, 4]],
                                      dtype=np.float32),
                             (1, 1, 4, 4))]

        layers = base.get_pooling_layer(PoolingLayer)(
            X, input_shapes, None)

        assert(len(layers) == 1)
        outpt = layers[0].forward_exec(inputs)

        expected_outpt = np.array([[[[5., 1., 8.],
                                     [5., 5., 8.],
                                     [9., 9., 4.]]]])
        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_maxpool_stride_2(self):

        iX = px.ops.input('input', shape=[1, 1, 4, 4])

        X = px.ops.pool2d(
            op_name='test_maxpool',
            input_layer=iX,
            pool_type='Max',
            pool_size=[2, 2],
            strides=[2, 2],
            padding=[0, 0],
            layout='NCHW'
        )

        input_shapes = {
            'input': TensorShape([1, 1, 4, 4])
        }
        inputs = [np.reshape(np.array([[1, 1, 0, 4], [5, 1, 0, 8],
                                       [3, 5, 1, 0], [1, 9, 3, 4]],
                                      dtype=np.float32),
                             (1, 1, 4, 4))]

        layers = base.get_pooling_layer(PoolingLayer)(
            X, input_shapes, None)

        assert(len(layers) == 1)
        outpt = layers[0].forward_exec(inputs)

        expected_outpt = np.array([[[[5., 8.], [9., 4.]]]])
        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_avgpool(self):

        iX = px.ops.input('input', shape=[1, 1, 4, 4])

        X = px.ops.pool2d(
            op_name='test_avgpool',
            input_layer=iX,
            pool_type='Avg',
            pool_size=[2, 2],
            strides=[2, 2],
            padding=[0, 0],
            layout='NCHW'
        )

        input_shapes = {
            'input': TensorShape([1, 1, 4, 4])
        }
        inputs = [np.reshape(np.array([[1, 1, 0, 4], [5, 1, 0, 8],
                                       [3, 5, 1, 0], [1, 9, 3, 4]],
                                      dtype=np.float32),
                             (1, 1, 4, 4))]

        layers = base.get_pooling_layer(PoolingLayer)(
            X, input_shapes, {})

        assert(len(layers) == 1)
        outpt = layers[0].forward_exec(inputs)

        expected_outpt = np.array([[[[2., 3.], [4.5, 2.]]]])
        np.testing.assert_array_equal(outpt, expected_outpt)

    def test_upsampling2d_nearest_neighbor(self):

        X = xlayer.XLayer(
            type=['Upsampling2D'],
            name='ups1',
            shapes=[1, 1, 4, 6],
            sizes=[32],
            bottoms=['in1'],
            tops=[],
            targets=[],
            attrs={
                'scale_h': 2,
                'scale_w': 3,
                'data_layout': 'NCHW',
                'method': 'nearest_neighbor',
                'align_corners': False
            }
        )

        input_shapes = {'in1': TensorShape([1, 1, 2, 2])}

        layers = X_2_TF['Upsampling2D'](X, input_shapes, {})
        assert len(layers) == 1

        inpt = np.array(
            [[1, 2],
             [3, 4]]).reshape((1, 1, 2, 2))
        inputs = [inpt]

        expected_outpt = np.array(
            [[1, 1, 1, 2, 2, 2],
             [1, 1, 1, 2, 2, 2],
             [3, 3, 3, 4, 4, 4],
             [3, 3, 3, 4, 4, 4]]
        ).reshape((1, 1, 4, 6))

        out = layers[0].forward_exec(inputs)

        assert out.shape == (1, 1, 4, 6)

        np.testing.assert_array_equal(out, expected_outpt)
