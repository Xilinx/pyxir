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
Module for testing the pyxir ONNX frontend


"""

import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l2_convolution as ol2c


class TestONNXL2Convolutions(unittest.TestCase):

    def test_eltwise_any_ops(self):

        any_ops = ['LRN']

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

            func = getattr(ol2c, any_op.lower())
            Xs = func(wrapped_node, params, xmap)

            assert len(Xs) == 1
            X = Xs[0]

            assert X.name == 'y'
            assert 'AnyOp' in X.type
            assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_avg_pool_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[0, 1, 0, 1],
            strides=[2, 2]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.avg_pool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Pooling' in X.type
        assert X.shapes.tolist() == [-1, 1, 2, 2]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 1], [0, 1]]
        assert X.attrs['strides'] == [2, 2]
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['type'] == 'Avg'

    def test_avg_pool_node_ceil_mode(self):
        x = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'AveragePool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[3, 3],
            strides=[2, 2],
            ceil_mode=True
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.avg_pool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Pooling' in X.type
        assert X.shapes.tolist() == [-1, 1, 2, 2]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['strides'] == [2, 2]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['type'] == 'Avg'

    def test_conv_node_0(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)
        W = np.array([[[[1, 1],
                        [1, 1]]],
                      [[[1, -1],
                        [1, 1]]]]).astype(np.float32)
        B = np.array([1, -1]).astype(np.float32)

        node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[1, 1, 0, 0]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        wX = xlf.get_xop_factory_func('Constant')('W', W, onnx_id='W')
        bX = xlf.get_xop_factory_func('Constant')('B', B, onnx_id='B')

        xmap = {'x': iX, 'W': wX, 'B': bX}
        params = {}

        Xs = ol2c.conv(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X, baX = Xs

        assert X.name == 'y_Conv'
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['padding'] == [(0, 0), (0, 0), (1, 0), (1, 0)]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['channels'] == [1, 2]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['groups'] == 1
        assert X.attrs['onnx_id'] == 'y'

        assert baX.name == 'y'
        assert baX.shapes == [-1, 2, 3, 3]
        assert baX.attrs['axis'] == 1
        assert baX.attrs['onnx_id'] == 'y'

    def test_conv_node_1(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)
        W = np.array([[[[1, 1],
                        [1, 1]]],
                      [[[1, -1],
                        [1, 1]]]]).astype(np.float32)
        B = np.array([1, -1]).astype(np.float32)

        node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[1, 1, 0, 0]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        wX = xlf.get_xop_factory_func('Constant')('W', W, onnx_id='W')
        bX = xlf.get_xop_factory_func('Constant')('B', B, onnx_id='B')

        xmap = {'x': iX, 'W': wX, 'B': bX}
        params = {}

        Xs = ol2c.conv(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X, baX = Xs

        assert X.name == 'y_Conv'
        assert X.shapes.tolist() == [-1, 2, 3, 3]
        assert X.attrs['padding'] == [(0, 0), (0, 0), (1, 0), (1, 0)]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['channels'] == [1, 2]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['groups'] == 1
        assert X.attrs['onnx_id'] == 'y'

        assert baX.name == 'y'
        assert baX.shapes == [-1, 2, 3, 3]
        assert baX.attrs['axis'] == 1
        assert baX.attrs['onnx_id'] == 'y'

    def test_depth_conv_node(self):
        x = np.ones((1,16,4,4)).astype(np.float32)
        W = np.ones((8,4,2,2)).astype(np.float32)
        B = np.ones((8,)).astype(np.float32)

        node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[1, 1, 0, 0],
            group=4
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        wX = xlf.get_xop_factory_func('Constant')('W', W, onnx_id='W')
        bX = xlf.get_xop_factory_func('Constant')('B', B, onnx_id='B')

        xmap = {'x': iX, 'W': wX, 'B': bX}
        params = {}

        Xs = ol2c.conv(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X, baX = Xs
        assert X.name == 'y_Conv'
        assert X.shapes.tolist() == [-1, 8, 4, 4]
        assert X.attrs['padding'] == [(0, 0), (0, 0), (1, 0), (1, 0)]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['channels'] == [16, 8]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['groups'] == 4
        assert X.attrs['onnx_id'] == 'y'

        assert baX.name == 'y'
        assert baX.shapes == [-1, 8, 4, 4]
        assert baX.attrs['axis'] == 1
        assert baX.attrs['onnx_id'] == 'y'

    def test_conv_transpose_node(self):
        x = np.zeros((1, 2, 3, 3))
        W = np.zeros((4, 2, 3, 3))
        B = np.array([1, -1]).astype(np.float32)

        node = onnx.helper.make_node(
            'ConvTranspose',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        wX = xlf.get_xop_factory_func('Constant')('W', W, onnx_id='W')
        bX = xlf.get_xop_factory_func('Constant')('B', B, onnx_id='B')

        xmap = {'x': iX, 'W': wX, 'B': bX}
        params = {}

        Xs = ol2c.conv_transpose(wrapped_node, params, xmap)

        assert len(Xs) == 2
        X, baX = Xs

        assert X.name == 'y_Conv'
        assert X.shapes.tolist() == [-1, 4, 5, 5]
        assert X.attrs['padding'] == [(0, 0), (0, 0), (0, 0), (0, 0)]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['dilation'] == [1, 1]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['channels'] == [2, 4]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['groups'] == 1
        assert X.attrs['onnx_id'] == 'y'

        assert baX.name == 'y'
        assert baX.shapes == [-1, 4, 5, 5]
        assert baX.attrs['axis'] == 1
        assert baX.attrs['onnx_id'] == 'y'

    def test_flatten_2_flatten(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Flatten',
            inputs=['x'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.flatten(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Flatten' in X.type
        assert X.shapes.tolist() == [-1, 9]
        assert X.attrs['onnx_id'] == 'y'

    def test_flatten_2_reshape(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],
                       [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Flatten',
            inputs=['x'],
            outputs=['y'],
            axis=2
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.flatten(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Reshape' in X.type
        assert X.shapes.tolist() == [-2, 9]
        assert X.attrs['shape'] == [-2, 9]
        assert X.attrs['onnx_id'] == 'y'

    def test_flatten_2_reshape_axis_0(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],
                       [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'Flatten',
            inputs=['x'],
            outputs=['y'],
            axis=0
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.flatten(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Reshape' in X.type
        assert X.shapes.tolist() == [1, -18]
        assert X.attrs['shape'] == [1, -18]
        assert X.attrs['onnx_id'] == 'y'

    def test_global_avg_pool_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]],
                       [[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'GlobalAveragePool',
            inputs=['x'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.global_avg_pool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Pooling' in X.type
        assert X.shapes.tolist() == [-1, 2, 1, 1]
        assert X.attrs['padding'] == [(0, 0), (0, 0), (0, 0), (0, 0)]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['pool_type'] == 'Avg'
        assert X.attrs['onnx_id'] == 'y'

    def test_max_pool_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'MaxPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[0, 1, 0, 1],
            strides=[1, 1]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.max_pool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Pooling' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 3]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 1], [0, 1]]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['type'] == 'Max'

    def test_max_roi_pool_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)
        a = np.array([[0, 0, 1, 0, 1], [0, 1, 2, 1, 2]])

        node = onnx.helper.make_node(
            'MaxRoiPool',
            inputs=['x', 'a'],
            outputs=['y'],
            pooled_shape=[2, 2]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        aX = xlf.get_xop_factory_func('Constant')('a', a)

        xmap = {'x': iX, 'a': aX}
        params = {}

        Xs = ol2c.max_roi_pool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [2, 1, 2, 2]

    def test_max_unpool_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        node = onnx.helper.make_node(
            'MaxUnPool',
            inputs=['x'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[0, 1, 0, 1],
            strides=[1, 1]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        xmap = {'x': iX}
        params = {}

        Xs = ol2c.max_unpool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_max_unpool_node_output_shape(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)
        z = np.array([-1, 1, 4, 4])

        node = onnx.helper.make_node(
            'MaxUnPool',
            inputs=['x', 'y', 'z'],
            outputs=['y'],
            kernel_shape=[2, 2],
            strides=[2, 2]
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        zX = xlf.get_xop_factory_func('Constant')('z', z)

        xmap = {'x': iX, 'z': zX}
        params = {}

        Xs = ol2c.max_unpool(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 4, 4]

    def test_pad(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)
        p = np.array([0, 0, 1, 1, 0, 0, 2, 3])
        pv = np.array([0])

        node = onnx.helper.make_node(
            'Pad',
            inputs=['x', 'p', 'pv'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        pX = xlf.get_xop_factory_func('Constant')('p', p)
        pvX = xlf.get_xop_factory_func('Constant')('pv', pv)

        xmap = {'x': iX, 'p': pX, 'pv': pvX}
        params = {}

        Xs = ol2c.pad(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Pad' in X.type
        assert X.shapes.tolist() == [-1, 1, 6, 7]

    def test_upsample_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

        node = onnx.helper.make_node(
            'Upsample',
            inputs=['x', 'scales'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')
        sX = xlf.get_xop_factory_func('Constant')('scales', scales)

        xmap = {'x': iX, 'scales': sX}
        params = {}

        Xs = ol2c.upsample(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Upsampling2D' in X.type
        assert X.shapes.tolist() == [-1, 1, 6, 9]
        assert X.attrs['scale_h'] == 2.
        assert X.attrs['scale_w'] == 3.
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['method'] == 'nearest_neighbor'

    def test_upsample7_node(self):
        x = np.array([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]]).astype(np.float32)

        scales = [1.0, 1.0, 2.0, 3.0]

        node = onnx.helper.make_node(
            'Upsample-7',
            inputs=['x'],
            outputs=['y'],
            scales=scales
        )

        wrapped_node = NodeWrapper(node)

        iX = xlf.get_xop_factory_func('Input')('x', list(x.shape),
                                               dtype='float32')

        xmap = {'x': iX}
        params = {}

        Xs = ol2c.upsample(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Upsampling2D' in X.type
        assert X.shapes.tolist() == [-1, 1, 6, 9]
        assert X.attrs['scale_h'] == 2.
        assert X.attrs['scale_w'] == 3.
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['method'] == 'nearest_neighbor'
