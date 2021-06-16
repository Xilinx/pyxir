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
Module for testing the XGraph layout transformation pass


"""

import os
import unittest

import numpy as np

from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.xgraph_factory import XGraphFactory

from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.transformers.layout_transformation_pass \
    import XGraphLayoutTransformationPass
from pyxir.target_registry import TargetRegistry, register_op_support_check

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestLayoutTransformationPass(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
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
        target_registry.register_target('npu_test',
                                        xgraph_optimizer,
                                        xgraph_quantizer,
                                        xgraph_compiler,
                                        xgraph_build_func)

        @register_op_support_check('npu_test', 'Convolution')
        def conv_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):

        target_registry = TargetRegistry()
        target_registry.unregister_target('npu_test')

    def test_simple(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['in1'],
                tops=[],
                layer=['conv1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            )
        ]
        xgraph = TestLayoutTransformationPass.xgraph_factory\
            .build_from_xlayer(net)

        layout_transform_pass = XGraphLayoutTransformationPass('NHWC')
        new_xgraph = layout_transform_pass.execute(xgraph)

        xlayers = new_xgraph.get_layers()
        # print(xlayers)
        assert len(new_xgraph) == 4
        assert xlayers[0].type[0] == 'Input'
        assert xlayers[1].type[0] == 'Transpose'
        assert xlayers[2].type[0] == 'Convolution'
        assert xlayers[3].type[0] == 'Transpose'

        assert xlayers[0].bottoms == []
        assert xlayers[0].tops == ['conv1_bottom_NCHW-NHWC']
        assert xlayers[0].shapes == [1, 1, 4, 4]
        assert xlayers[1].bottoms == ['in1']
        assert xlayers[1].tops == ['conv1']
        assert xlayers[1].shapes == [1, 4, 4, 1]
        assert xlayers[2].bottoms == ['conv1_bottom_NCHW-NHWC']
        assert xlayers[2].tops == ['conv1_top_NHWC-NCHW']
        assert xlayers[2].shapes == [1, 3, 3, 2]
        assert xlayers[3].bottoms == ['conv1']
        assert xlayers[3].tops == []
        assert xlayers[3].shapes == [1, 2, 3, 3]

        # NCHW -- NHWC
        assert xlayers[1].attrs['axes'] == [0, 2, 3, 1]
        # NHWC -- NCHW
        assert xlayers[3].attrs['axes'] == [0, 3, 1, 2]

        assert xlayers[2].attrs['data_layout'] == 'NHWC'

    def test_complex(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['in1'],
                tops=['pool1'],
                layer=['conv1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['pool1'],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=[],
                tops=['in2_transpose'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='in2_transpose',
                type=['Transpose'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=['in2'],
                tops=['conv2'],
                layer=['in2_transpose'],
                attrs={'axes': [0, 3, 1, 2]},
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['in2_transpose'],
                tops=['concat1'],
                layer=['conv2'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 4, 2, 2],
                sizes=[16],
                bottoms=['pool1', 'conv2'],
                tops=['concat1_transpose'],
                layer=['concat1'],
                attrs={
                    'axis': 1
                },
                targets=[]
            ),
            XLayer(
                name='concat1_transpose',
                type=['Transpose'],
                shapes=[1, 2, 2, 4],
                sizes=[16],
                bottoms=['concat1'],
                tops=['dense1'],
                layer=['concat1_transpose'],
                attrs={'axes': [0, 2, 3, 1]},
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[],
                bottoms=['concat1_transpose'],
                tops=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['dense1'],
                targets=[]
            )
        ]
        xgraph = TestLayoutTransformationPass.xgraph_factory\
            .build_from_xlayer(net)

        layout_transform_pass = XGraphLayoutTransformationPass('NHWC')
        new_xgraph = layout_transform_pass.execute(xgraph)

        xlayers = new_xgraph.get_layers()
        # print(xlayers)
        # print(len(xlayers))
        assert len(new_xgraph) == 8

        assert xlayers[0].type[0] == 'Input'
        assert xlayers[0].name == 'in1'
        assert xlayers[0].bottoms == []
        assert xlayers[0].tops == ['conv1_bottom_NCHW-NHWC']
        assert xlayers[0].shapes == [1, 1, 4, 4]
        
        assert xlayers[1].type[0] == 'Transpose'
        assert xlayers[1].name == 'conv1_bottom_NCHW-NHWC'
        assert xlayers[1].bottoms == ['in1']
        assert xlayers[1].tops == ['conv1']
        assert xlayers[1].shapes == [1, 4, 4, 1]
        assert xlayers[1].attrs['axes'] == [0, 2, 3, 1]

        assert xlayers[2].type[0] == 'Convolution'
        assert xlayers[2].name == 'conv1'
        assert xlayers[2].bottoms == ['conv1_bottom_NCHW-NHWC']
        assert xlayers[2].tops == ['pool1']
        assert xlayers[2].shapes == [1, 3, 3, 2]
        assert xlayers[2].attrs['data_layout'] == 'NHWC'

        assert xlayers[3].type[0] == 'Pooling'
        assert xlayers[3].name == 'pool1'
        assert xlayers[3].bottoms == ['conv1']
        assert xlayers[3].tops == ['concat1']
        assert xlayers[3].shapes == [1, 2, 2, 2]
        assert xlayers[3].attrs['data_layout'] == 'NHWC'

        assert xlayers[4].type[0] == 'Input'
        assert xlayers[4].name == 'in2'
        assert xlayers[4].bottoms == []
        assert xlayers[4].tops == ['conv2']
        assert xlayers[4].shapes == [1, 4, 4, 1]

        assert xlayers[5].type[0] == 'Convolution'
        assert xlayers[5].name == 'conv2'
        assert xlayers[5].bottoms == ['in2']
        assert xlayers[5].tops == ['concat1']
        assert xlayers[5].shapes == [1, 2, 2, 2]
        assert xlayers[5].attrs['data_layout'] == 'NHWC'

        assert xlayers[6].type[0] == 'Concat'
        assert xlayers[6].name == 'concat1'
        assert xlayers[6].bottoms == ['pool1', 'conv2']
        assert xlayers[6].tops == ['dense1']
        assert xlayers[6].shapes == [1, 2, 2, 4]
        assert xlayers[6].attrs['axis'] == 3

        assert xlayers[7].type[0] == 'Dense'
        assert xlayers[7].name == 'dense1'
        assert xlayers[7].bottoms == ['concat1']
        assert xlayers[7].tops == []
        assert xlayers[7].shapes == [1, 20]

    def test_target(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['in1'],
                tops=['pool1'],
                layer=['conv1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[],
                target='test'
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['pool1'],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=[],
                tops=['in2_transpose'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='in2_transpose',
                type=['Transpose'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=['in2'],
                tops=['conv2'],
                layer=['in2_transpose'],
                attrs={'axes': [0, 3, 1, 2]},
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['in2_transpose'],
                tops=['concat1'],
                layer=['conv2'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[],
                target='test'
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 4, 2, 2],
                sizes=[16],
                bottoms=['pool1', 'conv2'],
                tops=['concat1_transpose'],
                layer=['concat1'],
                attrs={
                    'axis': 1
                },
                targets=[]
            ),
            XLayer(
                name='concat1_transpose',
                type=['Transpose'],
                shapes=[1, 2, 2, 4],
                sizes=[16],
                bottoms=['concat1'],
                tops=['dense1'],
                layer=['concat1_transpose'],
                attrs={'axes': [0, 2, 3, 1]},
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[],
                bottoms=['concat1_transpose'],
                tops=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['dense1'],
                targets=[]
            )
        ]
        xgraph = TestLayoutTransformationPass.xgraph_factory\
            .build_from_xlayer(net)

        layout_transform_pass = XGraphLayoutTransformationPass('NHWC',
                                                               target='test')
        new_xgraph = layout_transform_pass.execute(xgraph)

        xlayers = new_xgraph.get_layers()
        # print(xlayers)
        # print(len(xlayers))
        assert len(new_xgraph) == 10

        assert xlayers[0].type[0] == 'Input'
        assert xlayers[0].name == 'in1'
        assert xlayers[0].bottoms == []
        assert xlayers[0].tops == ['conv1_bottom_NCHW-NHWC']
        assert xlayers[0].shapes == [1, 1, 4, 4]

        assert xlayers[1].type[0] == 'Transpose'
        assert xlayers[1].name == 'conv1_bottom_NCHW-NHWC'
        assert xlayers[1].bottoms == ['in1']
        assert xlayers[1].tops == ['conv1']
        assert xlayers[1].shapes == [1, 4, 4, 1]
        assert xlayers[1].attrs['axes'] == [0, 2, 3, 1]

        assert xlayers[2].type[0] == 'Convolution'
        assert xlayers[2].name == 'conv1'
        assert xlayers[2].bottoms == ['conv1_bottom_NCHW-NHWC']
        assert xlayers[2].tops == ['conv1_top_NHWC-NCHW']
        assert xlayers[2].shapes == [1, 3, 3, 2]
        assert xlayers[2].attrs['data_layout'] == 'NHWC'
        assert xlayers[2].attrs['padding'] == [[0, 0], [1, 1], [1, 1], [0, 0]]

        assert xlayers[3].type[0] == 'Transpose'
        assert xlayers[3].name == 'conv1_top_NHWC-NCHW'
        assert xlayers[3].bottoms == ['conv1']
        assert xlayers[3].tops == ['pool1']
        assert xlayers[3].shapes == [1, 2, 3, 3]
        assert xlayers[3].attrs['axes'] == (0, 3, 1, 2)

        assert xlayers[4].type[0] == 'Pooling'
        assert xlayers[4].name == 'pool1'
        assert xlayers[4].bottoms == ['conv1_top_NHWC-NCHW']
        assert xlayers[4].tops == ['0_split_concat1_transpose']
        assert xlayers[4].shapes == [1, 2, 2, 2]
        assert xlayers[4].attrs['data_layout'] == 'NCHW'
        assert xlayers[4].attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]

        assert xlayers[5].type[0] == 'Transpose'
        assert xlayers[5].name == '0_split_concat1_transpose'
        assert xlayers[5].bottoms == ['pool1']
        assert xlayers[5].tops == ['concat1']
        assert xlayers[5].shapes == [1, 2, 2, 2]
        assert xlayers[5].attrs['axes'] == [0, 2, 3, 1]

        assert xlayers[6].type[0] == 'Input'
        assert xlayers[6].name == 'in2'
        assert xlayers[6].bottoms == []
        assert xlayers[6].tops == ['conv2']
        assert xlayers[6].shapes == [1, 4, 4, 1]

        assert xlayers[7].type[0] == 'Convolution'
        assert xlayers[7].name == 'conv2'
        assert xlayers[7].bottoms == ['in2']
        assert xlayers[7].tops == ['concat1']
        assert xlayers[7].shapes == [1, 2, 2, 2]
        assert xlayers[7].attrs['data_layout'] == 'NHWC'

        assert xlayers[8].type[0] == 'Concat'
        assert xlayers[8].name == 'concat1'
        assert xlayers[8].bottoms == ['0_split_concat1_transpose', 'conv2']
        assert xlayers[8].tops == ['dense1']
        assert xlayers[8].shapes == [1, 2, 2, 4]
        assert xlayers[8].attrs['axis'] == 3

        assert xlayers[9].type[0] == 'Dense'
        assert xlayers[9].name == 'dense1'
        assert xlayers[9].bottoms == ['concat1']
        assert xlayers[9].tops == []
        assert xlayers[9].shapes == [1, 20]
