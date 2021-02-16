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

"""Module for testing the subgraph quantization flow"""

import os
import sys
import json
import unittest
import numpy as np
import logging

from pyxir import partition
from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.graph.layer.xlayer import XLayer, ConvData, BatchData, ScaleData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.quantization.base_subgraph_quantizer import \
    XGraphBaseSubgraphQuantizer

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

# logger = logging.getLogger('pyxir')
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.DEBUG)


try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Quantization Tensorflow related test because Tensorflow"
                            " is not available")


class BaseSubgraphQuantizerTest(XGraphBaseSubgraphQuantizer):

    def quantize_subgraph(self, xgraph, inputs, input_names, output_names):

        self.test_inputs = {
            xgraph.get_name(): inputs
        }


class TestBaseSubgraphQuantizer(unittest.TestCase):

    xgraph_factory = XGraphFactory()

    @classmethod
    def setUpClass(cls):

        def xgraph_build_func(xgraph):
            raise NotImplementedError("")

        def xgraph_optimizer(xgraph):
            raise NotImplementedError("")

        def xgraph_quantizer(xgraph):
            raise NotImplementedError("")

        def xgraph_compiler(xgraph):
            raise NotImplementedError("")

        target_registry = TargetRegistry()
        target_registry.register_target('test',
                                        xgraph_optimizer,
                                        xgraph_quantizer,
                                        xgraph_compiler,
                                        xgraph_build_func)

        @register_op_support_check('test', 'Pooling')
        def pooling_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test', 'Concat')
        def concat_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):

        target_registry = TargetRegistry()
        target_registry.unregister_target('test')

    def test_one_subgraph(self):

        W = np.reshape(
            np.array([
                [[1, 1],
                 [0, 1]],
                [[3, 4],
                 [-1, 0]]], dtype=np.float32),
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
                tops=['pool2'],
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
                name='pool2',
                type=['Pooling'],
                shapes=[1, 2, 1, 1],
                sizes=[2],
                bottoms=['pool1'],
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
                    'data_layout': 'NCHW',
                    'pool_type': 'Avg'
                }
            )
        ]
        xgraph = TestBaseSubgraphQuantizer.xgraph_factory\
            .build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])

        assert len(p_xgraph.get_layer_names()) == 4
        assert p_xgraph.get_subgraph_names() == ['xp0']

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ['Input']
        assert p_xlayers[1].type[0] in ['Convolution']
        assert p_xlayers[2].type[0] in ['Pooling']
        assert p_xlayers[3].type[0] in ['Pooling']

        inputs = np.reshape(np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.float32), (1, 1, 4, 4))

        def inputs_func(iter):
            return {'in1': inputs}

        quantizer = BaseSubgraphQuantizerTest(
            xgraph=p_xgraph,
            inputs_func=inputs_func
        )
        quantizer.quantize()

        assert 'xp0' in quantizer.test_inputs
        assert 'xinput0' in quantizer.test_inputs['xp0']

        expected = np.reshape(np.array([
            [[4, 4, 4],
             [4, 4, 4],
             [4, 4, 4]],
            [[5, 5, 5],
             [5, 5, 5],
             [5, 5, 5]]]),
            (1, 2, 3, 3))
        np.testing.assert_array_equal(
            quantizer.test_inputs['xp0']['xinput0'], expected)

    def test_multiple_subgraph(self):

        W = np.reshape(
            np.array([
                [[1, 1],
                 [0, 1]],
                [[3, 4],
                 [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.], dtype=np.float32)

        W2 = np.reshape(
            np.array([
                [[1, 1],
                 [0, 1]],
                [[3, 4],
                 [-1, 0]]], dtype=np.float32),
            (1, 2, 2, 2))
        B2 = np.array([-1.], dtype=np.float32)

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
                tops=['conv2'],
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
                name='conv2',
                type=['Convolution'],
                shapes=[-1, 1, 1, 1],
                sizes=[18],
                bottoms=['pool1'],
                tops=['pool2'],
                layer=['conv2'],
                data=ConvData(W2, B2),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'shape': [1, 1, 1, 1],
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            ),
            XLayer(
                name='pool2',
                type=['Pooling'],
                shapes=[1, 1, 1, 1],
                sizes=[2],
                bottoms=['conv2'],
                tops=[],
                layer=['pool2'],
                targets=[],
                attrs={
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'kernel_size': [1, 1],
                    'insize': [1, 1],
                    # HW
                    'outsize': [1, 1],
                    'data_layout': 'NCHW',
                    'pool_type': 'Avg'
                }
            )
        ]
        xgraph = TestBaseSubgraphQuantizer.xgraph_factory\
            .build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])

        assert(len(p_xgraph.get_layer_names()) == 5)
        assert(set(p_xgraph.get_subgraph_names()) == set(['xp0']))

        p_xlayers = p_xgraph.get_layers()
        assert(p_xlayers[0].type[0] in ['Input'])
        assert(p_xlayers[1].type[0] in ['Convolution'])
        assert(p_xlayers[2].type[0] in ['Pooling'])
        assert(p_xlayers[3].type[0] in ['Convolution'])
        assert(p_xlayers[4].type[0] in ['Pooling'])

        inputs = np.reshape(np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.float32), (1, 1, 4, 4))

        def inputs_func(iter):
            return {'in1': inputs}

        quantizer = BaseSubgraphQuantizerTest(
            xgraph=p_xgraph,
            inputs_func=inputs_func
        )
        quantizer.quantize()

        assert 'xp0' in quantizer.test_inputs
        assert 'xinput0' in quantizer.test_inputs['xp0']

        expected = np.reshape(
            np.array([[
                [[4, 4, 4],
                 [4, 4, 4],
                 [4, 4, 4]],
                [[5, 5, 5],
                 [5, 5, 5],
                 [5, 5, 5]]]]), (1, 2, 3, 3))
        np.testing.assert_array_equal(
            quantizer.test_inputs['xp0']['xinput0'], expected)

        # assert('xp1' in quantizer.test_inputs)
        # assert('xinput1' in quantizer.test_inputs['xp1'])

        # expected = np.reshape(np.array([[[41]]]), (1, 1, 1, 1))
        # np.testing.assert_array_equal(
        #     quantizer.test_inputs['xp1']['xinput1'], expected)
