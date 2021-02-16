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

"""Module for testing the default quantization flow"""

import os
import sys
import json
import math
import logging
import unittest
import numpy as np

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Quantization Tensorflow related test because Tensorflow"
                            " is not available")

from pyxir.quantization.mse_quantization.mse_threshold_quantizer import \
    XGraphMSEThresholdQuantizer
from pyxir.graph.layer.xlayer import XLayer, ConvData, BatchData, ScaleData
from pyxir.graph.xgraph_factory import XGraphFactory

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

# logger = logging.getLogger('pyxir')
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.DEBUG)


class TestMSEQuantizer(unittest.TestCase):

    xgraph_factory = XGraphFactory()

    def test_simple(self):

        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.], dtype=np.float32)

        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[-1, 2, 3, 3],
                sizes=[18],
                bottoms=['in1'],
                tops=['pool1'],
                layer=['conv2d0'],
                data=ConvData(W, B),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'shape': [1, 2, 3, 3],
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv1'],
                tops=[],
                layer=['pool1'],
                targets=[],
                attrs={
                    'kernel_size': [2, 2],
                    'insize': [3, 3],
                    # HW
                    'outsize': [2, 2],
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'pool_type': 'Max'
                }
            )
        ]
        xgraph = TestMSEQuantizer.xgraph_factory.build_from_xlayer(net)

        def inputs_func(iter):
            inputs = np.reshape(
                np.array([
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]], dtype=np.float32),
                (1, 1, 4, 4))

            return {'in1': inputs}

        quantizer = XGraphMSEThresholdQuantizer(xgraph,
                                                inputs_func,
                                                work_dir=FILE_PATH)
        quantizer._quantize(xgraph, subgraphs_only=False)

        assert 'xgraph' in quantizer._quant_layers

        assert len(quantizer._quant_layers['xgraph']) == 2
        assert ('conv1', 'Convolution', None) in\
            quantizer._quant_layers['xgraph']

        # assert quantizer._quant_param.th_layer_out['in1'] == [1.]

        assert quantizer._quant_param.th_layer_in['conv1'] == [1.]
        np.testing.assert_array_equal(
            quantizer._quant_param.th_params['conv1'], np.array([1., 4.]))
        assert quantizer._quant_param.th_layer_out['conv1'][0] <= 5.
        assert quantizer._quant_param.th_layer_out['conv1'][0] >= 0.

        assert quantizer._quant_param.th_layer_in['pool1'][0] <= 5.
        assert quantizer._quant_param.th_layer_in['pool1'][0] >= 0.
        assert quantizer._quant_param.th_layer_out['pool1'][0] <= 5.
        assert quantizer._quant_param.th_layer_out['pool1'][0] >= 0.

        # # Test json saving
        quant_file = os.path.join(FILE_PATH, 'quant1.json')
        quantizer._quant_param.save_to_dpu_v1_json(
            quantizer._quant_layers['xgraph'], quant_file)

        with open(quant_file) as f:
            qp_d = json.load(f)
            network = qp_d['network']

        assert len(network) == 2

        assert network[0]['name'] == 'conv1'
        assert network[0]['th_layer_in'] == 1.0
        assert network[0]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['conv1'][0]
        assert network[0]['th_params'] ==\
            list(quantizer._quant_param.th_params['conv1'])

        assert network[1]['name'] == 'pool1'
        assert network[1]['th_layer_in'] == \
            quantizer._quant_param.th_layer_out['conv1'][0]
        assert network[1]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['conv1'][0]

        os.remove(quant_file)

    def test_two_quant_parts(self):

        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.], dtype=np.float32)

        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[-1, 2, 3, 3],
                sizes=[18],
                bottoms=['in1'],
                tops=['pool1'],
                layer=['conv2d0'],
                data=ConvData(W, B),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'shape': [1, 2, 3, 3],
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv1'],
                tops=['t1'],
                layer=['pool1'],
                targets=[],
                attrs={
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'kernel_size': [2, 2],
                    'insize': [3, 3],
                    # HW
                    'outsize': [2, 2],
                    'data_layout': 'NCHW',
                    'pool_type': 'Max'
                }
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1'],
                tops=['pool2'],
                layer=['t1'],
                targets=[],
                attrs={
                    'axes': [0, 2, 3, 1]
                }
            ),
            XLayer(
                name='pool2',
                type=['Pooling'],
                shapes=[1, 1, 1, 2],
                sizes=[2],
                bottoms=['t1'],
                tops=[],
                layer=['pool2'],
                targets=[],
                attrs={
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'kernel_size': [2, 2],
                    'insize': [2, 2],
                    # HW
                    'outsize': [1, 1],
                    'data_layout': 'NHWC',
                    'pool_type': 'Avg'
                }
            )
        ]
        xgraph = TestMSEQuantizer.xgraph_factory.build_from_xlayer(net)

        def inputs_func(iter):
            inputs = np.reshape(
                np.array([
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1]], dtype=np.float32),
                (1, 1, 4, 4))

            return {'in1': inputs}

        quantizer = XGraphMSEThresholdQuantizer(xgraph,
                                                inputs_func,
                                                work_dir=FILE_PATH)
        quantizer._quantize(xgraph, subgraphs_only=False)

        assert 'xgraph' in quantizer._quant_layers

        assert len(quantizer._quant_layers['xgraph']) == 3
        assert ('conv1', 'Convolution', None) in\
            quantizer._quant_layers['xgraph']

        assert quantizer._quant_param.th_layer_in['conv1'] == [1.]
        np.testing.assert_array_equal(
            quantizer._quant_param.th_params['conv1'], np.array([1., 4.]))
        assert quantizer._quant_param.th_layer_out['conv1'][0] <= 5.
        assert quantizer._quant_param.th_layer_out['conv1'][0] >= 0.

        assert quantizer._quant_param.th_layer_in['pool1'][0] <= 5.
        assert quantizer._quant_param.th_layer_in['pool1'][0] >= 0.
        assert quantizer._quant_param.th_layer_out['pool1'][0] <= 5.
        assert quantizer._quant_param.th_layer_out['pool1'][0] >= 0.

        assert 't1' not in quantizer._quant_param.th_layer_in
        assert 't1' not in quantizer._quant_param.th_layer_out

        assert quantizer._quant_param.th_layer_in['pool2'][0] <= 5.
        assert quantizer._quant_param.th_layer_in['pool2'][0] >= 0.
        assert quantizer._quant_param.th_layer_out['pool2'][0] <= 5.
        assert quantizer._quant_param.th_layer_out['pool2'][0] >= 0.

        # # Test json saving
        quant_file = os.path.join(FILE_PATH, 'quant1.json')
        quantizer._quant_param.save_to_dpu_v1_json(
            quantizer._quant_layers['xgraph'], quant_file)

        with open(quant_file) as f:
            qp_d = json.load(f)
            network = qp_d['network']

        assert len(network) == 3

        assert network[0]['name'] == 'conv1'
        assert network[0]['th_layer_in'] == 1.0
        assert network[0]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['conv1'][0]
        assert network[0]['th_params'] ==\
            list(quantizer._quant_param.th_params['conv1'])

        assert network[1]['name'] == 'pool1'
        assert network[1]['th_layer_in'] == \
            quantizer._quant_param.th_layer_in['pool1'][0]
        assert network[1]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['pool1'][0]

        assert network[2]['name'] == 'pool2'
        assert network[2]['th_layer_in'] == \
            quantizer._quant_param.th_layer_in['pool2'][0]
        assert network[2]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['pool2'][0]

        os.remove(quant_file)

    def test_resnet_block(self):

        W = np.reshape(
            np.array([[[1, 0, 1], [1, 0, 1], [1, 0, 1]]], dtype=np.float32),
            (1, 1, 3, 3))
        B = np.array([0.], dtype=np.float32)

        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['add1', 'conv1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=['in1'],
                tops=['add1'],
                layer=['conv1'],
                data=ConvData(W, B),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'shape': [1, 1, 4, 4],
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=['in1', 'conv1'],
                tops=['add1_relu'],
                layer=['add1'],
                targets=[],
                attrs={}
            ),
            XLayer(
                name='add1_relu',
                type=['ReLU'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=['add1'],
                tops=[],
                layer=['add1_relu'],
                targets=[],
                attrs={}
            )
        ]
        xgraph = TestMSEQuantizer.xgraph_factory.build_from_xlayer(net)

        def inputs_func(iter):
            inputs = np.reshape(np.array([
                    [1, 3, -1, -11],
                    [3, 1, -1, 0],
                    [1, 4, -3, -3],
                    [1, 1, -1, -1]
                ], dtype=np.float32), (1, 1, 4, 4))

            return {'in1': inputs}

        quantizer = XGraphMSEThresholdQuantizer(xgraph,
                                                inputs_func,
                                                work_dir=FILE_PATH)
        quantizer._quantize(xgraph, subgraphs_only=False)

        assert 'xgraph' in quantizer._quant_layers

        assert len(quantizer._quant_layers['xgraph']) == 2
        assert ('conv1', 'Convolution', None) in\
            quantizer._quant_layers['xgraph']

        assert quantizer._quant_param.th_layer_in['conv1'][0] <= 11.
        np.testing.assert_array_equal(
            quantizer._quant_param.th_params['conv1'], np.array([1.]))
        assert quantizer._quant_param.th_layer_out['conv1'][0] <= 11.
        assert quantizer._quant_param.th_layer_out['conv1'][0] >= 0.

        assert quantizer._quant_param.th_layer_in['add1'][0] <= 11.
        assert quantizer._quant_param.th_layer_in['add1'][0] >= 0.
        assert quantizer._quant_param.th_layer_out['add1'][0] <= 14.
        assert quantizer._quant_param.th_layer_out['add1'][0] >= 0.

        # # Test json saving
        quant_file = os.path.join(FILE_PATH, 'quant3.json')
        quantizer._quant_param.save_to_dpu_v1_json(
            quantizer._quant_layers['xgraph'], quant_file)

        with open(quant_file) as f:
            qp_d = json.load(f)
            network = qp_d['network']

        assert len(network) == 2

        assert network[0]['name'] == 'conv1'
        assert network[0]['th_layer_in'] == \
            quantizer._quant_param.th_layer_in['conv1'][0]
        assert network[0]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['conv1'][0]
        assert network[0]['th_params'] ==\
            list(quantizer._quant_param.th_params['conv1'])

        assert network[1]['name'] == 'add1'
        assert network[1]['th_layer_in'] == \
            quantizer._quant_param.th_layer_in['add1'][0]
        assert network[1]['th_layer_out'] == \
            quantizer._quant_param.th_layer_out['add1'][0]

        os.remove(quant_file)

    def test_inception_block(self):

        W1 = np.reshape(
            np.array([[[1, 0, 1], [1, 0, 1], [1, 0, 1]]], dtype=np.float32),
                    (1, 1, 3, 3))
        B1 = np.array([0.], dtype=np.float32)
        W2 = np.reshape(
            np.array([[[1, 1, 0], [1, 1, 0], [1, 1, 0]]], dtype=np.float32),
            (1, 1, 3, 3))
        B2 = np.array([0.], dtype=np.float32)

        gamma = np.array([2.])
        beta = np.array([1.])

        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['scale1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='scale1',
                type=['Scale'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=['in1'],
                tops=['conv1', 'conv2'],
                layer=['scale1'],
                targets=[],
                data=ScaleData(gamma, beta),
                attrs={'axis': 1}
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=['scale1'],
                tops=['concat1'],
                layer=['conv1'],
                data=ConvData(W1, B1),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'shape': [1, 1, 4, 4],
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[-1, 1, 4, 4],
                sizes=[16],
                bottoms=['scale1'],
                tops=['concat1'],
                layer=['conv2'],
                data=ConvData(W2, B2),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'shape': [1, 1, 4, 4],
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[-1, 2, 4, 4],
                sizes=[16],
                bottoms=['conv1', 'conv2'],
                tops=[],
                layer=['concat1'],
                targets=[],
                attrs={
                    'axis': 1
                }
            )
        ]
        xgraph = TestMSEQuantizer.xgraph_factory.build_from_xlayer(net)

        def inputs_func(iter):
            inputs = np.reshape(np.array([
                    [1, 3, -1, -11],
                    [3, 1, -1, 0],
                    [1, 4, -3, -3],
                    [1, 1, -1, -1]
                ], dtype=np.float32), (1, 1, 4, 4))

            return {'in1': inputs}

        quantizer = XGraphMSEThresholdQuantizer(xgraph,
                                                inputs_func,
                                                work_dir=FILE_PATH)
        q_xgraph = quantizer.quantize(subgraphs_only=False)

        assert 'xgraph' in quantizer._quant_layers
        assert len(quantizer._quant_layers['xgraph']) == 4

        # assert(('scale1', 'Scale', None) in \
        #    quantizer._quant_layers['xgraph'])
        assert ('conv1', 'Convolution', None) in\
            quantizer._quant_layers['xgraph']
        assert ('conv2', 'Convolution', None) in\
            quantizer._quant_layers['xgraph']
        assert ('concat1', 'Concat', None) in\
            quantizer._quant_layers['xgraph']

        assert quantizer._quant_param.th_layer_in['scale1'][0] <= 11.
        assert quantizer._quant_param.th_layer_in['scale1'][0] >= 0.
        assert quantizer._quant_param.th_layer_out['scale1'][0] <= 22.
        assert quantizer._quant_param.th_layer_out['scale1'][0] >= 0.

        assert quantizer._quant_param.th_layer_in['conv1'] ==\
            quantizer._quant_param.th_layer_out['scale1']
        np.testing.assert_array_equal(
            quantizer._quant_param.th_params['conv1'], np.array([1.]))
        # NOTE: Conv2d does not take over threshold from subqequent concat
        #   layer because it's expected that a scaling layer will be inserted
        assert quantizer._quant_param.th_layer_out['conv1'][0] <= 22.

        assert quantizer._quant_param.th_layer_in['conv2'][0] ==\
            quantizer._quant_param.th_layer_out['scale1']
        np.testing.assert_array_equal(
            quantizer._quant_param.th_params['conv2'], np.array([1.]))
        assert quantizer._quant_param.th_layer_out['conv2'][0] <= 33.

        # print("TEST")
        # print(quantizer._quant_param.th_layer_in['concat1'])
        # print(quantizer._quant_param.th_layer_out['conv2'])
        assert math.isclose(
            quantizer._quant_param.th_layer_in['concat1'],
            quantizer._quant_param.th_layer_out['conv2'],
            rel_tol=1e-4)
        # assert quantizer._quant_param.th_layer_in['concat1'] ==\
        #     quantizer._quant_param.th_layer_out['conv2']
        assert quantizer._quant_param.th_layer_out['concat1'] ==\
            quantizer._quant_param.th_layer_in['concat1']

        quant_file = os.path.join(FILE_PATH, 'xgraph_quant.json')
        with open(quant_file) as f:
            qp_d = json.load(f)
            network = qp_d['network']

        assert network[0]['name'] == 'scale1'
        assert network[0]['th_layer_in'] ==\
            quantizer._quant_param.th_layer_in['scale1'][0]
        assert network[0]['th_layer_out'] ==\
            quantizer._quant_param.th_layer_out['scale1'][0]

        assert network[1]['name'] == 'conv1'
        assert network[1]['th_layer_in'] ==\
            quantizer._quant_param.th_layer_in['conv1'][0]
        # NOTE: adjustment for different scaling of concat input layers ! conv2
        assert network[1]['th_layer_out'] ==\
            quantizer._quant_param.th_layer_out['conv2'][0]
        assert network[1]['th_params'] == [1.0]

        assert network[2]['name'] == 'conv2'
        assert network[2]['th_layer_in'] ==\
            quantizer._quant_param.th_layer_in['conv2'][0]
        assert network[2]['th_layer_out'] ==\
            quantizer._quant_param.th_layer_out['conv2'][0]
        assert network[2]['th_params'] == [1.0]

        assert network[3]['name'] == 'concat1'
        assert network[3]['th_layer_in'] ==\
            quantizer._quant_param.th_layer_in['concat1'][0]
        assert network[3]['th_layer_out'] ==\
            quantizer._quant_param.th_layer_out['concat1'][0]

        os.remove(quant_file)
