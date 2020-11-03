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
Module for testing the XOp factory and property functionality


"""

import unittest
import numpy as np
import pyxir as px

from pyxir.graph.layer.xlayer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph import ops


class TestL2Convolution(unittest.TestCase):

    def test_nn_batch_flatten_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 1, 1, 4],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[]
        )

        sX = px.ops.batch_flatten('flatten1', iX)

        assert sX.type[0] == 'Flatten'
        assert sX.shapes == [1, 4]
        assert sX.sizes == [4]
        assert sX.attrs == {}

    def test_batchnorm_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[]
        )

        mX = XLayer(
            type=['Constant'],
            name='mu',
            shapes=[2],
            sizes=[2],
            data=[np.array([.5, 1.])],
            bottoms=[],
            tops=[],
            targets=[]
        )

        sqX = XLayer(
            type=['Constant'],
            name='sigma_square',
            shapes=[2],
            sizes=[2],
            data=[np.array([1., 2.])],
            bottoms=[],
            tops=[],
            targets=[]
        )

        gX = XLayer(
            type=['Constant'],
            name='gamma',
            shapes=[2],
            sizes=[2],
            data=[np.array([1., 2.])],
            bottoms=[],
            tops=[],
            targets=[]
        )

        bX = XLayer(
            type=['Constant'],
            name='beta',
            shapes=[2],
            sizes=[2],
            data=[np.array([1., -2.])],
            bottoms=[],
            tops=[],
            targets=[]
        )

        bX = px.ops.batch_norm('bn1', iX, mX, sqX, gX, bX,
                               axis=1, epsilon=1e-5)

        assert bX.type[0] == 'BatchNorm'
        assert bX.attrs['axis'] == 1
        assert bX.attrs['epsilon'] == 1e-5

        np.testing.assert_array_equal(bX.data.gamma, np.array([1., 2.]))
        np.testing.assert_array_equal(bX.data.beta, np.array([1., -2.]))
        np.testing.assert_array_equal(bX.data.mu, np.array([.5, 1.]))
        np.testing.assert_array_equal(
            bX.data.sigma_square, np.array([1., 2.]))

        from pyxir.graph.ops.l2_convolution import \
            batchnorm_transpose_transform

        batchnorm_transpose_transform(bX, axes=[0, 2, 3, 1])

        assert bX.type[0] == 'BatchNorm'
        assert bX.shapes == [1, 4, 4, 2]
        assert bX.attrs['axis'] == 3
        assert bX.attrs['epsilon'] == 1e-5

    def test_convolution_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 2, 3, 3],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[]
        )

        kX = XLayer(
            type=['Constant'],
            name='kernel',
            shapes=[4, 2, 3, 3],
            sizes=[54],
            data=[np.ones((4, 2, 3, 3), dtype=np.float32)],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('Convolution')(
            op_name='conv1',
            kernel_size=[3, 3],
            strides=[1, 1],
            padding_hw=[1, 1],
            dilation=[1, 1],
            groups=1,
            channels=4,
            data_layout='NCHW',
            kernel_layout='OIHW',
            input_layer=iX,
            weights_layer=kX
        )

        assert X.type[0] == 'Convolution'
        assert X.shapes == [1, 4, 3, 3]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['shape'] == [1, 4, 3, 3]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['groups'] == 1
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['channels'] == [2, 4]

        np.testing.assert_array_equal(
            X.data.weights, np.ones((4, 2, 3, 3), dtype=np.float32))
        np.testing.assert_array_equal(
            X.data.biases, np.zeros((4), dtype=np.float32))

        from pyxir.graph.ops.l2_convolution import \
            conv2d_layout_transform

        conv2d_layout_transform(X, target_layout='NHWC')

        assert X.type[0] == 'Convolution'
        assert X.shapes == [1, 3, 3, 4]
        assert X.attrs['data_layout'] == 'NHWC'
        assert X.attrs['padding'] == [[0, 0], [1, 1], [1, 1], [0, 0]]

    def test_convolution_layer_tfl(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 3, 3, 2],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[]
        )

        kX = XLayer(
            type=['Constant'],
            name='kernel',
            shapes=[4, 3, 3, 2],
            sizes=[54],
            data=[np.transpose(np.ones((4, 2, 3, 3), dtype=np.float32), (0, 2, 3, 1))],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('Convolution')(
            op_name='conv1',
            kernel_size=[3, 3],
            strides=[1, 1],
            padding_hw=[1, 1],
            dilation=[1, 1],
            groups=1,
            channels=4,
            data_layout='NHWC',
            kernel_layout='OHWI',
            input_layer=iX,
            weights_layer=kX
        )

        assert X.type[0] == 'Convolution'
        assert X.shapes == [1, 3, 3, 4]
        assert X.attrs['padding'] == [[0, 0], [1, 1], [1, 1], [0, 0]]
        assert X.attrs['data_layout'] == 'NHWC'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['shape'] == [1, 3, 3, 4]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['groups'] == 1
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['channels'] == [2, 4]

        np.testing.assert_array_equal(
            X.data.weights, np.ones((4, 2, 3, 3), dtype=np.float32))
        np.testing.assert_array_equal(
            X.data.biases, np.zeros((4), dtype=np.float32))

        from pyxir.graph.ops.l2_convolution import \
            conv2d_layout_transform

        conv2d_layout_transform(X, target_layout='NCHW')

        assert X.type[0] == 'Convolution'
        assert X.shapes == [1, 4, 3, 3]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]

    def test_depthwise_convolution_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 8, 3, 3],
            sizes=[72],
            bottoms=[],
            tops=[],
            targets=[]
        )

        kX = XLayer(
            type=['Constant'],
            name='kernel',
            shapes=[4, 2, 3, 3],
            sizes=[54],
            data=[np.ones((4, 2, 3, 3), dtype=np.float32)],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('Convolution')(
            op_name='conv1',
            kernel_size=[3, 3],
            strides=[1, 1],
            padding_hw=[1, 1],
            dilation=[1, 1],
            groups=4,
            channels=4,
            data_layout='NCHW',
            kernel_layout='OIHW',
            input_layer=iX,
            weights_layer=kX
        )

        assert X.type[0] == 'Convolution'
        assert X.shapes == [1, 4, 3, 3]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['shape'] == [1, 4, 3, 3]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['groups'] == 4
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['channels'] == [8, 4]

        np.testing.assert_array_equal(
            X.data.weights, np.ones((4, 2, 3, 3), dtype=np.float32))
        np.testing.assert_array_equal(
            X.data.biases, np.zeros((4), dtype=np.float32))

        from pyxir.graph.ops.l2_convolution import \
            conv2d_layout_transform

        conv2d_layout_transform(X, target_layout='NHWC')

        assert X.type[0] == 'Convolution'
        assert X.shapes == [1, 3, 3, 4]
        assert X.attrs['data_layout'] == 'NHWC'
        assert X.attrs['padding'] == [[0, 0], [1, 1], [1, 1], [0, 0]]

    def test_conv2d_transpose_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 2, 3, 3],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[]
        )

        kX = XLayer(
            type=['Constant'],
            name='kernel',
            shapes=[4, 2, 3, 3],
            sizes=[54],
            data=[np.ones((4, 2, 3, 3), dtype=np.float32)],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('Conv2DTranspose')(
            op_name='conv1',
            kernel_size=[3, 3],
            strides=[1, 1],
            padding_hw=[0, 0],
            dilation=[1, 1],
            groups=1,
            channels=4,
            data_layout='NCHW',
            kernel_layout='OIHW',
            input_layer=iX,
            weights_layer=kX
        )

        assert X.type[0] == 'Conv2DTranspose'
        assert X.shapes == [1, 4, 5, 5]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['shape'] == [1, 4, 5, 5]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['groups'] == 1
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['channels'] == [2, 4]

        np.testing.assert_array_equal(
            X.data.weights, np.ones((4, 2, 3, 3), dtype=np.float32))
        np.testing.assert_array_equal(
            X.data.biases, np.zeros((4), dtype=np.float32))

        from pyxir.graph.ops.l2_convolution import \
            conv2d_transpose_layout_transform

        conv2d_transpose_layout_transform(X, target_layout='NHWC')

        assert X.type[0] == 'Conv2DTranspose'
        assert X.shapes == [1, 5, 5, 4]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['data_layout'] == 'NHWC'

    def test_global_pooling_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 2, 7, 7],
            sizes=[98],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('GlobalPooling')(
            op_name='gp1',
            pool_type='Max',
            layout='NCHW',
            input_layer=iX
        )

        assert X.type[0] == 'Pooling'
        assert X.shapes == [1, 2, 1, 1]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['insize'] == [7, 7]
        assert X.attrs['outsize'] == [1, 1]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['kernel_size'] == [7, 7]
        assert X.attrs['pool_type'] == 'Max'

        from pyxir.graph.ops.l2_convolution import \
            pooling_layout_transform

        pooling_layout_transform(X, target_layout='NHWC')

        assert X.type[0] == 'Pooling'
        assert X.shapes == [1, 1, 1, 2]
        assert X.attrs['data_layout'] == 'NHWC'

    def test_pad_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 2, 7, 7],
            sizes=[98],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('Pad')(
            op_name='pad1',
            padding=[[0, 0], [0, 0], [1, 0], [1, 0]],
            pad_value=0,
            input_layer=iX
        )

        assert X.type[0] == 'Pad'
        assert X.shapes == [1, 2, 8, 8]
        assert X.sizes == [128]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 0], [1, 0]]

        from pyxir.graph.ops.l2_convolution import \
            padding_transpose_transform

        padding_transpose_transform(X, axes=(0, 2, 3, 1))

        assert X.type[0] == 'Pad'
        assert X.shapes == [1, 8, 8, 2]
        assert X.attrs['padding'] == [[0, 0], [1, 0], [1, 0], [0, 0]]

    def test_pooling_layer(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 2, 5, 5],
            sizes=[50],
            bottoms=[],
            tops=[],
            targets=[]
        )

        X = xlf.get_xop_factory_func('Pooling')(
            op_name='pool1',
            input_layer=iX,
            pool_type='Avg',
            pool_size=[3, 3],
            strides=[2, 2],
            padding=[1, 1],
            layout='NCHW',
            ceil_mode=True,
            count_include_pad=True
        )

        assert X.type[0] == 'Pooling'
        assert X.shapes == [1, 2, 3, 3]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs['insize'] == [5, 5]
        assert X.attrs['outsize'] == [3, 3]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['strides'] == [2, 2]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['pool_type'] == 'Avg'

        from pyxir.graph.ops.l2_convolution import \
            pooling_layout_transform

        pooling_layout_transform(X, target_layout='NHWC')

        assert X.type[0] == 'Pooling'
        assert X.shapes == [1, 3, 3, 2]
        assert X.attrs['padding'] == [[0, 0], [1, 1], [1, 1], [0, 0]]
        assert X.attrs['data_layout'] == 'NHWC'

    def test_nn_upsampling2d(self):

        iX = XLayer(
            type=['Input'],
            name='in1',
            shapes=[1, 4, 2, 2],
            sizes=[16],
            bottoms=[],
            tops=[],
            targets=[]
        )

        sX = xlf.get_xop_factory_func('Upsampling2D')(
            'ups1',
            [iX],
            scale_h=3,
            scale_w=2,
            data_layout='NCHW',
            method='nearest_neighbor',
            align_corners=False
        )

        assert sX.type[0] == 'Upsampling2D'
        assert sX.shapes == [1, 4, 6, 4]
        assert sX.sizes == [96]
        assert sX.attrs['scale_h'] == 3
        assert sX.attrs['scale_w'] == 2
        assert sX.attrs['data_layout'] == 'NCHW'
        assert sX.attrs['method'] == 'nearest_neighbor'
        assert sX.attrs['align_corners'] is False

        from pyxir.graph.ops.l2_convolution import \
            upsampling2d_layout_transform

        upsampling2d_layout_transform(sX, target_layout='NHWC')

        assert sX.type[0] == 'Upsampling2D'
        assert sX.shapes == [1, 6, 4, 4]
        assert sX.attrs['data_layout'] == 'NHWC'
