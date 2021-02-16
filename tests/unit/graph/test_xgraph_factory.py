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

"""Module for testing the xgraph functionality"""

import unittest

import numpy as np

# ! Important for device registration
import pyxir

from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.xgraph import XGraph


class TestXGraphFactory(unittest.TestCase):

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

        @register_op_support_check('test', 'Convolution')
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test', 'Pooling')
        def pooling_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):
        target_registry = TargetRegistry()
        target_registry.unregister_target('test')

    def test_xgraph_factory(self):

        xlayers = [
            XLayer(
                name='in1',
                type=['Input'],
                bottoms=[],
                tops=['conv1'],
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                bottoms=[],
                tops=['add1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                bottoms=['in1'],
                tops=['add1'],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0., 1.], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                bottoms=['conv1', 'in2'],
                tops=['conv2', 'pool1'],
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                bottoms=['add1'],
                tops=['add2'],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0., 1.], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                bottoms=['add1'],
                tops=['add2'],
                targets=[]
            ),
            XLayer(
                name='add2',
                type=['Eltwise'],
                bottoms=['conv2', 'pool1'],
                tops=[],
                targets=[]
            )
        ]
        xgraph = TestXGraphFactory.xgraph_factory.build_from_xlayer(xlayers)

        # GENERAL
        assert len(xgraph) == 7
        assert len(xgraph.get_layer_names()) == 7
        assert xgraph.get_layer_names() == \
            ['in1', 'conv1', 'in2', 'add1', 'conv2', 'pool1', 'add2']
        assert len(xgraph.get_output_names()) == 1
        assert len(xgraph.get_input_names()) == 2

        # DEVICES
        xlayers = xgraph.get_layers()
        # assert set(xlayers[0].targets) == set(['cpu'])
        # assert set(xlayers[1].targets) == set(['cpu', 'test'])
        # assert set(xlayers[2].targets) == set(['cpu'])
        # assert set(xlayers[3].targets) == set(['cpu'])
        # assert set(xlayers[4].targets) == set(['cpu', 'test'])
        # assert set(xlayers[5].targets) == set(['cpu', 'test'])
        # assert set(xlayers[6].targets) == set(['cpu'])

        # Bottoms / tops
        assert xgraph.get_top_layers('in1')[0].name == 'conv1'
        assert len(xgraph.get_bottom_layers('in1')) == 0

        assert xgraph.get_top_layers('in2')[0].name == 'add1'
        assert len(xgraph.get_bottom_layers('in2')) == 0

        assert len(xgraph.get_bottom_layers('conv1')) == 1
        assert len(xgraph.get_top_layers('conv1')) == 1
        assert xgraph.get_top_layers('conv1')[0].name == 'add1'
        assert xgraph.get_bottom_layers('conv1')[0].name == 'in1'

        assert len(xgraph.get_bottom_layers('add1')) == 2
        assert len(xgraph.get_top_layers('add1')) == 2
        assert xgraph.get_bottom_layers('add1')[0].name == 'conv1'
        assert xgraph.get_top_layers('add1')[0].name == 'conv2'
