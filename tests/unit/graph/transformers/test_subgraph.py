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

"""Module for testing the xgraph partitioning functionality"""

import os
import sys
import unittest
import logging

import numpy as np

# ! Important for device registration
import pyxir

from pyxir import partition
from pyxir.graph.layer.xlayer import XLayer, ConvData, BatchData
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.graph.transformers import subgraph

# logger = logging.getLogger('pyxir')
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.DEBUG)


class TestSubgraphBuildFunc(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    target_registry = TargetRegistry()

    @classmethod
    def setUpClass(cls):

        def xgraph_build_func_simple(xgraph):
            return subgraph.xgraph_build_func(
                xgraph=xgraph,
                target='test_simple',
                xtype='TEST_SIMPLE',
                layout='NCHW',
            )

        def xgraph_build_func(xgraph):
            return subgraph.xgraph_build_func(
                xgraph=xgraph,
                target='test',
                xtype='TEST',
                layout='NHWC'
            )

        def xgraph_optimizer(xgraph):
            raise NotImplementedError("")

        def xgraph_quantizer(xgraph):
            raise NotImplementedError("")

        def xgraph_compiler(xgraph):
            raise NotImplementedError("")

        TestSubgraphBuildFunc.target_registry.register_target(
            'test',
            xgraph_optimizer,
            xgraph_quantizer,
            xgraph_compiler,
            xgraph_build_func)
        TestSubgraphBuildFunc.target_registry.register_target(
            'test_simple',
            xgraph_optimizer,
            xgraph_quantizer,
            xgraph_compiler,
            xgraph_build_func_simple)

        @register_op_support_check('test', 'Convolution')
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test', 'Pooling')
        def pooling_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test', 'Concat')
        def concat_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test_simple', 'Convolution')
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test_simple', 'Pooling')
        def pooling_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test_simple', 'Concat')
        def concat_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):

        TestSubgraphBuildFunc.target_registry.unregister_target('test')
        TestSubgraphBuildFunc.target_registry.unregister_target('test_simple')

    def test_basic(self):
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
                name='in2',
                type=['Input'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=[],
                tops=['add1'],
                layer=['in2'],
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
                tops=['add1'],
                layer=['pool1'],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1', 'in2'],
                tops=[],
                layer=['add1'],
                targets=[]
            )
        ]
        xgraph = TestSubgraphBuildFunc.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test_simple'])
        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test_simple')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        # print(layers)
        assert len(dpu_xgraph) == 5

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'TEST_SIMPLE'
        assert layers[2].type[0] == 'TupleGetItem'
        assert layers[3].type[0] == 'Input'
        assert layers[4].type[0] == 'Eltwise'

        assert layers[0].bottoms == []
        assert layers[0].tops == ['xp0']

        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['pool1']
        assert layers[1].attrs['target'] == 'test_simple'
        assert layers[1].attrs['input_names'] == ['xinput0']
        assert layers[1].attrs['output_names'] == ['pool1']
        assert layers[1].attrs['input_layers']['xinput0'] == ['conv1']
        assert layers[1].attrs['output_layers']['pool1'] == ['pool1']
        assert layers[1].attrs['__bottom_tensors'] == {'xinput0': ['in1']}
        assert layers[1].attrs['__top_tensors'] == {'pool1': ['add1']}

        assert layers[2].bottoms == ['xp0']
        assert layers[2].tops == ['add1']

        assert layers[3].bottoms == []
        assert layers[3].tops == ['add1']

        assert layers[4].bottoms == ['pool1', 'in2']
        assert layers[4].tops == []

    def test_two_partitions_interrupt(self):
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
                tops=['pool1', 'bn1'],
                layer=['conv1'],
                targets=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                }
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 4, 3, 3],
                sizes=[36],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['pool1'],
                targets=[],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                }
            ),
            XLayer(
                name='bn1',
                type=['BatchNorm'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['bn1'],
                data=BatchData(np.array([1, 1]), np.array([0, 0]),
                               np.array([1, 1]), np.array([0, 0])),
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 6, 3, 3],
                sizes=[54],
                bottoms=['pool1', 'bn1'],
                tops=['conv2'],
                layer=['concat1'],
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 10, 2, 2],
                sizes=[40],
                bottoms=['concat1'],
                tops=[],
                layer=['conv2'],
                targets=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                }
            )
        ]
        xgraph = TestSubgraphBuildFunc.xgraph_factory\
            .build_from_xlayer(net)

        p_xgraph = partition(xgraph, ['test_simple'])

        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test_simple')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        assert len(dpu_xgraph) == 7

        assert layers[0].type[0] == 'Input'
        assert layers[0].bottoms == []
        assert layers[0].tops == ['xp0']

        assert layers[1].type[0] == 'TEST_SIMPLE'
        assert layers[1].shapes == [[1, 2, 3, 3], [1, 4, 3, 3]]
        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['conv1', 'pool1']
        assert layers[1].attrs['input_names'] == ['xinput0']
        assert set(layers[1].attrs['output_names']) == set(['pool1', 'conv1'])
        assert layers[1].attrs['target'] == 'test_simple'
        assert layers[1].attrs['__bottom_tensors'] == {'xinput0': ['in1']}
        assert layers[1].attrs['orig_bottom_tensors'] == {'xinput0': ['in1']}
        assert layers[1].attrs['__top_tensors'] == \
            {'conv1': ['bn1'], 'pool1': ['concat1']}
        assert layers[1].attrs['orig_top_tensors'] == \
            {'conv1': ['bn1'], 'pool1': ['concat1']}

        assert layers[2].type[0] == 'TupleGetItem'
        assert layers[2].name == 'pool1'
        assert layers[2].bottoms == ['xp0']
        assert layers[2].shapes == [1, 4, 3, 3]
        assert layers[2].tops == ['concat1']
        assert layers[2].attrs['index'] == 1

        assert layers[3].type[0] == 'TupleGetItem'
        assert layers[3].name == 'conv1'
        assert layers[3].bottoms == ['xp0']
        assert layers[3].shapes == [1, 2, 3, 3]
        assert layers[3].tops == ['bn1']
        assert layers[3].attrs['index'] == 0

        assert layers[4].type[0] == 'BatchNorm'
        assert layers[4].name == 'bn1'
        assert layers[4].bottoms == ['conv1']
        assert layers[4].shapes == [1, 2, 3, 3]
        assert layers[4].tops == ['concat1']

        assert layers[5].type[0] == 'Concat'
        assert layers[5].name == 'concat1'
        assert layers[5].bottoms == ['pool1', 'bn1']
        assert layers[5].shapes == [1, 6, 3, 3]
        assert layers[5].tops == ['conv2']

        assert layers[6].type[0] == 'Convolution'
        assert layers[6].name == 'conv2'
        assert layers[6].bottoms == ['concat1']
        assert layers[6].shapes == [1, 10, 2, 2]
        assert layers[6].tops == []

    def test_basic_diff_layout(self):
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
                name='in2',
                type=['Input'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=[],
                tops=['add1'],
                layer=['in2'],
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
                tops=['add1'],
                layer=['pool1'],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1', 'in2'],
                tops=[],
                layer=['add1'],
                targets=[]
            )
        ]
        xgraph = TestSubgraphBuildFunc.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])
        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        # print(layers)
        assert(len(dpu_xgraph) == 6)

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].name == 'in1')
        assert(layers[0].bottoms == [])
        assert(layers[0].tops == ['conv1_bottom_NCHW-NHWC'])

        assert(layers[1].type[0] == 'Transpose')
        assert(layers[1].name == 'conv1_bottom_NCHW-NHWC')
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['xp0'])

        assert(layers[2].type[0] == 'TEST')
        assert(layers[2].bottoms == ['conv1_bottom_NCHW-NHWC'])
        assert(layers[2].tops == ['pool1'])
        assert(layers[2].attrs['target'] == 'test')
        assert(layers[2].attrs['input_names'] == ['xinput0'])
        assert(layers[2].attrs['output_names'] == ['pool1'])
        assert(layers[2].attrs['input_layers']['xinput0'] == ['conv1'])
        assert(layers[2].attrs['output_layers']['pool1'] == ['pool1'])
        assert(layers[2].attrs['__bottom_tensors'] ==
               {'xinput0': ['conv1_bottom_NCHW-NHWC']})
        assert(layers[2].attrs['orig_bottom_tensors'] == {'xinput0': ['in1']})
        assert(layers[2].attrs['__top_tensors'] ==
               {'pool1': ['pool1_top_NHWC-NCHW']})
        assert(layers[2].attrs['orig_top_tensors'] == {'pool1': ['add1']})

        assert(layers[3].type[0] == 'TupleGetItem')
        assert(layers[3].bottoms == ['xp0'])
        assert(layers[3].tops == ['add1'])
        assert layers[3].attrs['transpose'] is True
        assert layers[3].attrs['axes'] == [0, 3, 1, 2]

        # assert(layers[4].type[0] == 'Transpose')
        # assert(layers[4].name == 'pool1_top_NHWC-NCHW')
        # assert(layers[4].bottoms == ['pool1'])
        # assert(layers[4].tops == ['add1'])
        # assert layers[4].attrs['axes'] == [0, 3, 1, 2]

        assert layers[4].type[0] == 'Input'
        assert layers[4].name == 'in2'
        assert layers[4].bottoms == []
        assert layers[4].tops == ['add1']

        assert layers[5].type[0] == 'Eltwise'
        assert layers[5].name == 'add1'
        assert layers[5].bottoms == ['pool1', 'in2']
        assert layers[5].tops == []

    def test_two_partition_inputs(self):
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
                name='in2',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv2'],
                layer=['in2'],
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
                name='conv2',
                type=['Convolution'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['in2'],
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
                tops=['dense1'],
                layer=['concat1'],
                attrs={
                    'axis': 1
                },
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[],
                bottoms=['concat1'],
                tops=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['dense1'],
                targets=[]
            )
        ]
        xgraph = TestSubgraphBuildFunc.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])
        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        assert len(dpu_xgraph) == 7

        assert layers[0].type[0] == 'Input'
        assert layers[0].name == 'in1'
        assert layers[0].bottoms == []
        assert layers[0].tops == ['conv1_bottom_NCHW-NHWC']
        assert layers[0].target == 'cpu'
        assert layers[0].subgraph is None

        assert layers[1].type[0] == 'Transpose'
        assert layers[1].name == 'conv1_bottom_NCHW-NHWC'
        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['xp2']
        assert layers[1].target == 'cpu'
        assert layers[1].subgraph is None

        assert layers[2].type[0] == 'Input'
        assert layers[2].name == 'in2'
        assert layers[2].bottoms == []
        assert layers[2].tops == ['conv2_bottom_NCHW-NHWC']
        assert layers[2].target == 'cpu'
        assert layers[2].subgraph is None

        assert layers[3].type[0] == 'Transpose'
        assert layers[3].name == 'conv2_bottom_NCHW-NHWC'
        assert layers[3].bottoms == ['in2']
        assert layers[3].tops == ['xp2']
        assert layers[3].target == 'cpu'
        assert layers[3].subgraph is None

        assert layers[4].type[0] == 'TEST'
        assert layers[4].name == 'xp2'
        assert layers[4].bottoms == ['conv1_bottom_NCHW-NHWC',
                                     'conv2_bottom_NCHW-NHWC']
        assert layers[4].tops == ['concat1']
        assert layers[4].attrs['target'] == 'test'
        assert layers[4].attrs['input_names'] == ['xinput0', 'xinput1']
        assert layers[4].attrs['output_names'] == ['concat1']
        assert layers[4].attrs['input_layers']['xinput0'] == ['conv1']
        assert layers[4].attrs['input_layers']['xinput1'] == ['conv2']
        assert layers[4].attrs['output_layers']['concat1'] == ['concat1']
        assert(layers[4].attrs['__bottom_tensors'] ==
               {'xinput0': ['conv1_bottom_NCHW-NHWC'],
                'xinput1': ['conv2_bottom_NCHW-NHWC']})
        assert(layers[4].attrs['orig_bottom_tensors'] ==
               {'xinput0': ['in1'],
                'xinput1': ['in2']})
        assert layers[4].attrs['__top_tensors'] == {
            'concat1':
                ['merge_pool1_top_NHWC-NCHW_conv2_top_NHWC-NCHW']
            }
        assert layers[4].attrs['orig_top_tensors'] == {
            'concat1': ['dense1']
            }
        assert layers[4].target == 'cpu'
        assert layers[4].subgraph is None

        assert layers[5].type[0] == 'TupleGetItem'
        assert layers[5].name == 'concat1'
        assert layers[5].bottoms == ['xp2']
        assert layers[5].tops == ['dense1']
        assert layers[5].target == 'cpu'
        assert layers[5].subgraph is None
        assert layers[5].attrs['transpose'] is True

        # assert layers[6].type[0] == 'Transpose'
        # assert layers[6].name ==\
        #     'merge_pool1_top_NHWC-NCHW_conv2_top_NHWC-NCHW'
        # assert layers[6].bottoms == ['concat1']
        # assert layers[6].tops == ['dense1']
        # assert layers[6].target == 'cpu'
        # assert layers[6].subgraph is None

        assert layers[6].type[0] == 'Dense'
        assert layers[6].name == 'dense1'
        assert layers[6].bottoms == ['concat1']
        assert layers[6].tops == []
        assert layers[6].target == 'cpu'
        assert layers[6].subgraph is None

    def test_two_partition_diff_layout(self):
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
                layer=['in2'],
                attrs={'axes': [0, 3, 1, 2]},
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
                layer=['concat1'],
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
        xgraph = TestSubgraphBuildFunc.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])
        p_xlayers = p_xgraph.get_layers()

        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        assert len(dpu_xgraph) == 6

        assert layers[0].type[0] == 'Input'
        assert layers[0].name == 'in1'
        assert layers[0].shapes == [1, 1, 4, 4]
        assert layers[0].bottoms == []
        assert layers[0].tops == ['conv1_bottom_NCHW-NHWC']
        assert layers[0].target == 'cpu'
        assert layers[0].subgraph is None

        assert layers[1].type[0] == 'Transpose'
        assert layers[1].name == 'conv1_bottom_NCHW-NHWC'
        assert layers[1].shapes == [1, 4, 4, 1]
        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['xp2']
        assert layers[1].target == 'cpu'
        assert layers[1].subgraph is None

        assert layers[2].type[0] == 'Input'
        assert layers[2].name == 'in2'
        assert layers[2].shapes == [1, 4, 4, 1]
        assert layers[2].bottoms == []
        assert layers[2].tops == ['xp2']
        assert layers[2].target == 'cpu'
        assert layers[2].subgraph is None

        assert layers[3].type[0] == 'TEST'
        assert layers[3].name == 'xp2'
        assert layers[3].shapes == [[1, 2, 2, 4]]
        assert layers[3].bottoms == ['conv1_bottom_NCHW-NHWC', 'in2']
        assert layers[3].tops == ['concat1']
        assert layers[3].target == 'cpu'
        assert layers[3].subgraph is None
        assert layers[3].attrs['target'] == 'test'
        assert layers[3].attrs['input_names'] == ['xinput0', 'xinput1']
        assert layers[3].attrs['output_names'] == ['concat1']
        assert layers[3].attrs['input_layers']['xinput0'] == ['conv1']
        assert layers[3].attrs['input_layers']['xinput1'] == ['conv2']
        assert layers[3].attrs['output_layers']['concat1'] == ['concat1']
        assert(layers[3].attrs['__bottom_tensors'] ==
               {'xinput0': ['conv1_bottom_NCHW-NHWC'],
                'xinput1': ['in2']})
        assert(layers[3].attrs['orig_bottom_tensors'] ==
               {'xinput0': ['in1'],
                'xinput1': ['in2']})
        assert layers[3].attrs['__top_tensors'] == {'concat1': ['dense1']}
        assert layers[3].attrs['orig_top_tensors'] == {'concat1': ['dense1']}

        assert layers[4].type[0] == 'TupleGetItem'
        assert layers[4].name == 'concat1'
        assert layers[4].shapes == [1, 2, 2, 4]
        assert layers[4].bottoms == ['xp2']
        assert layers[4].tops == ['dense1']
        assert layers[4].target == 'cpu'
        assert layers[4].subgraph is None

        assert layers[5].type[0] == 'Dense'
        assert layers[5].name == 'dense1'
        assert layers[5].shapes == [1, 20]
        assert layers[5].bottoms == ['concat1']
        assert layers[5].tops == []
        assert layers[5].target == 'cpu'
        assert layers[5].subgraph is None

    def test_two_partition_inputs_complex(self):
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
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv2'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['in2'],
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
                tops=['dense1'],
                layer=['concat1'],
                attrs={
                    'axis': 1
                },
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[],
                bottoms=['concat1'],
                tops=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['dense1'],
                targets=[]
            )
        ]
        xgraph = TestSubgraphBuildFunc.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])
        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        assert len(dpu_xgraph) == 7

        assert layers[0].type[0] == 'Input'
        assert layers[0].name == 'in1'
        assert layers[0].shapes == [1, 1, 4, 4]
        assert layers[0].bottoms == []
        assert layers[0].tops == ['conv1_bottom_NCHW-NHWC']
        assert layers[0].target == 'cpu'
        assert layers[0].subgraph is None

        assert layers[1].type[0] == 'Transpose'
        assert layers[1].name == 'conv1_bottom_NCHW-NHWC'
        assert layers[1].shapes == [1, 4, 4, 1]
        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['xp2']
        assert layers[1].target == 'cpu'
        assert layers[1].subgraph is None

        assert layers[2].type[0] == 'Input'
        assert layers[2].name == 'in2'
        assert layers[2].shapes == [1, 1, 4, 4]
        assert layers[2].bottoms == []
        assert layers[2].tops == ['conv2_bottom_NCHW-NHWC']
        assert layers[2].target == 'cpu'
        assert layers[2].subgraph is None

        assert layers[3].type[0] == 'Transpose'
        assert layers[3].name == 'conv2_bottom_NCHW-NHWC'
        assert layers[3].shapes == [1, 4, 4, 1]
        assert layers[3].bottoms == ['in2']
        assert layers[3].tops == ['xp2']
        assert layers[3].target == 'cpu'
        assert layers[3].subgraph is None

        assert layers[4].type[0] == 'TEST'
        assert layers[4].name == 'xp2'
        assert layers[4].shapes == [[1, 2, 2, 4]]
        assert layers[4].bottoms == ['conv1_bottom_NCHW-NHWC',
                                     'conv2_bottom_NCHW-NHWC']
        assert layers[4].tops == ['concat1']
        assert layers[4].target == 'cpu'
        assert layers[4].subgraph is None
        assert layers[4].tops == ['concat1']
        assert layers[4].attrs['target'] == 'test'
        assert layers[4].attrs['input_names'] == ['xinput0', 'xinput1']
        assert layers[4].attrs['output_names'] == ['concat1']
        assert layers[4].attrs['input_layers']['xinput0'] == ['conv1']
        assert layers[4].attrs['input_layers']['xinput1'] == ['conv2']
        assert layers[4].attrs['output_layers']['concat1'] == ['concat1']
        assert(layers[4].attrs['__bottom_tensors'] ==
               {'xinput0': ['conv1_bottom_NCHW-NHWC'],
                'xinput1': ['conv2_bottom_NCHW-NHWC']})
        assert(layers[4].attrs['orig_bottom_tensors'] ==
               {'xinput0': ['in1'],
                'xinput1': ['in2']})
        assert layers[4].attrs['__top_tensors'] ==\
            {'concat1':
                ['merge_pool1_top_NHWC-NCHW_conv2_top_NHWC-NCHW']}
        assert layers[4].attrs['orig_top_tensors'] ==\
            {'concat1': ['dense1']}

        assert layers[5].type[0] == 'TupleGetItem'
        assert layers[5].name == 'concat1'
        assert layers[5].shapes == [1, 4, 2, 2]
        assert layers[5].bottoms == ['xp2']
        assert layers[5].tops == ['dense1']
        assert layers[5].attrs['transpose'] is True
        assert layers[5].attrs['axes'] == [0, 3, 1, 2]

        # assert layers[6].type[0] == 'Transpose'
        # assert layers[6].name ==\
        #     'merge_pool1_top_NHWC-NCHW_conv2_top_NHWC-NCHW'
        # assert layers[6].shapes == [1, 4, 2, 2]
        # assert layers[6].bottoms == ['concat1']
        # assert layers[6].tops == ['dense1']

        assert layers[6].type[0] == 'Dense'
        assert layers[6].name == 'dense1'
        assert layers[6].shapes == [1, 20]
        assert layers[6].bottoms == ['concat1']
        assert layers[6].tops == []

    def test_inception_like_block(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['concat1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['concat1'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 2, 4, 4],
                sizes=[32],
                bottoms=['in1', 'in2'],
                tops=['conv1', 'conv2'],
                layer=['concat1'],
                attrs={
                    'axis': 1
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 4, 3, 3],
                sizes=[],
                bottoms=['concat1'],
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
                shapes=[1, 4, 2, 2],
                sizes=[],
                bottoms=['conv1'],
                tops=['concat2'],
                layer=['pool1'],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 4, 2, 2],
                sizes=[],
                bottoms=['concat1'],
                tops=['concat2'],
                layer=['conv2'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[]
            ),
            XLayer(
                name='concat2',
                type=['Concat'],
                shapes=[1, 8, 2, 2],
                sizes=[32],
                bottoms=['pool1', 'conv2'],
                tops=['dense1'],
                layer=['concat2'],
                attrs={
                    'axis': 1
                },
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[20],
                bottoms=['concat2'],
                tops=[],
                layer=['dense1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                targets=[]
            )
        ]
        xgraph = TestSubgraphBuildFunc.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ['test'])
        dpu_xgraph = TestSubgraphBuildFunc.target_registry\
            .get_target_build_func('test')(p_xgraph)

        layers = dpu_xgraph.get_layers()
        assert len(dpu_xgraph) == 7

        assert layers[0].type[0] == 'Input'
        assert layers[0].name == 'in1'
        assert layers[0].shapes == [1, 1, 4, 4]
        assert layers[0].bottoms == []
        assert layers[0].tops ==\
            ['0_split_conv1_bottom_NCHW-NHWC_conv2_bottom_NCHW-NHWC']
        assert layers[0].target == 'cpu'
        assert layers[0].subgraph is None

        assert layers[1].type[0] == 'Transpose'
        assert layers[1].name ==\
            '0_split_conv1_bottom_NCHW-NHWC_conv2_bottom_NCHW-NHWC'
        assert layers[1].shapes == [1, 4, 4, 1]
        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['xp0']
        assert layers[1].target == 'cpu'
        assert layers[1].subgraph is None

        assert layers[2].type[0] == 'Input'
        assert layers[2].name == 'in2'
        assert layers[2].shapes == [1, 1, 4, 4]
        assert layers[2].bottoms == []
        assert layers[2].tops ==\
            ['1_split_conv1_bottom_NCHW-NHWC_conv2_bottom_NCHW-NHWC']
        assert layers[2].target == 'cpu'
        assert layers[2].subgraph is None

        assert layers[3].type[0] == 'Transpose'
        assert layers[3].name ==\
            '1_split_conv1_bottom_NCHW-NHWC_conv2_bottom_NCHW-NHWC'
        assert layers[3].shapes == [1, 4, 4, 1]
        assert layers[3].bottoms == ['in2']
        assert layers[3].tops == ['xp0']
        assert layers[3].target == 'cpu'
        assert layers[3].subgraph is None

        assert layers[4].type[0] == 'TEST'
        assert layers[4].name == 'xp0'
        assert layers[4].shapes == [[1, 2, 2, 8]]
        assert layers[4].bottoms ==\
            ['0_split_conv1_bottom_NCHW-NHWC_conv2_bottom_NCHW-NHWC',
             '1_split_conv1_bottom_NCHW-NHWC_conv2_bottom_NCHW-NHWC']
        assert layers[4].tops == ['concat2']
        assert layers[4].target == 'cpu'
        assert layers[4].subgraph is None
        assert layers[4].tops == ['concat2']
        assert layers[4].attrs['target'] == 'test'
        assert layers[4].attrs['input_names'] == ['xinput0', 'xinput1']
        assert layers[4].attrs['output_names'] == ['concat2']
        assert layers[4].attrs['input_layers']['xinput0'] == ['concat1']
        assert layers[4].attrs['input_layers']['xinput1'] == ['concat1']
        assert layers[4].attrs['output_layers']['concat2'] == ['concat2']
        assert(layers[4].attrs['__bottom_tensors'] ==
               {'xinput0': ['0_split_conv1_bottom_NCHW-NHWC_conv2_bottom'
                            '_NCHW-NHWC'],
                'xinput1': ['1_split_conv1_bottom_NCHW-NHWC_conv2_bottom'
                            '_NCHW-NHWC']})
        assert(layers[4].attrs['orig_bottom_tensors'] ==
               {'xinput0': ['in1'],
                'xinput1': ['in2']})
        assert layers[4].attrs['__top_tensors'] ==\
            {'concat2':
                ['merge_pool1_top_NHWC-NCHW_conv2_top_NHWC-NCHW']}
        assert layers[4].attrs['orig_top_tensors'] ==\
            {'concat2': ['dense1']}

        assert layers[5].type[0] == 'TupleGetItem'
        assert layers[5].name == 'concat2'
        assert layers[5].shapes == [1, 8, 2, 2]
        assert layers[5].bottoms == ['xp0']
        assert layers[5].tops == ['dense1']

        # assert layers[6].type[0] == 'Transpose'
        # assert layers[6].name ==\
        #     'merge_pool1_top_NHWC-NCHW_conv2_top_NHWC-NCHW'
        # assert layers[6].shapes == [1, 8, 2, 2]
        # assert layers[6].bottoms == ['concat2']
        # assert layers[6].tops == ['dense1']

        assert layers[6].type[0] == 'Dense'
        assert layers[6].name == 'dense1'
        assert layers[6].shapes == [1, 20]
        assert layers[6].bottoms == ['concat2']
        assert layers[6].tops == []
