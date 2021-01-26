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
import unittest

import numpy as np

# ! Important for device registration
import pyxir

from pyxir.graph.layer.xlayer import XLayer, ConvData, BatchData
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.target_registry import TargetRegistry, register_op_support_check


class TestXGraphPartitioner(unittest.TestCase):

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

        @register_op_support_check('test', 'Concat')
        def concat_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):

        target_registry = TargetRegistry()
        target_registry.unregister_target('test')

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
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv1'],
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
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(xgraph, ['test'])

        assert len(p_xgraph.get_layer_names()) == 5
        assert p_xgraph.get_subgraph_names() == ['xp0']

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ['Input']
        assert p_xlayers[1].type[0] in ['Convolution']
        assert p_xlayers[2].type[0] in ['Pooling']
        assert p_xlayers[3].type[0] in ['Input']
        assert p_xlayers[4].type[0] in ['Eltwise']

        assert p_xlayers[0].target == 'cpu'
        assert p_xlayers[1].target == 'test'
        assert p_xlayers[2].target == 'test'
        assert p_xlayers[3].target == 'cpu'
        assert p_xlayers[4].target == 'cpu'

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == 'xp0'
        assert p_xlayers[2].subgraph == 'xp0'
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == 'xp0'
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert xp0.bottoms == ['in1']
        assert xp0.tops == ['add1']
        assert xp0.shapes == [[1, 2, 2, 2]]
        assert xp0.sizes == [8]

        assert len(xp0_xgraph) == 3
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == 'Input'
        assert xp0_layers[0].layer[0] == 'conv1'
        assert xp0_layers[1].type[0] == 'Convolution'
        assert xp0_layers[2].type[0] == 'Pooling'

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ['conv1']
        assert xp0_layers[1].bottoms == ['xinput0']
        assert xp0_layers[1].tops == ['pool1']
        assert xp0_layers[2].bottoms == ['conv1']
        assert xp0_layers[2].tops == []

    def test_complete_partition(self):
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
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv1'],
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
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert len(p_xgraph.get_layer_names()) == 3
        assert p_xgraph.get_subgraph_names() == ['xp0']

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ['Input']
        assert p_xlayers[1].type[0] in ['Convolution']
        assert p_xlayers[2].type[0] in ['Pooling']

        assert p_xlayers[0].target == 'cpu'
        assert p_xlayers[1].target == 'test'
        assert p_xlayers[2].target == 'test'

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == 'xp0'
        assert p_xlayers[2].subgraph == 'xp0'

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == 'xp0'
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert xp0.bottoms == ['in1']
        assert xp0.tops == []
        assert xp0.shapes == [[1, 2, 2, 2]]
        assert xp0.sizes == [8]
        assert xp0.attrs['target'] == 'test'
        assert xp0.attrs['__bottom_tensors'] == {'xinput0': ['in1']}
        assert xp0.attrs['orig_bottom_tensors'] == {'xinput0': ['in1']}
        assert xp0.attrs['__top_tensors'] == {'pool1': []}
        assert xp0.attrs['orig_top_tensors'] == {'pool1': []}

        assert len(xp0_xgraph) == 3
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == 'Input'
        assert xp0_layers[0].layer[0] == 'conv1'
        assert xp0_layers[1].type[0] == 'Convolution'
        assert xp0_layers[2].type[0] == 'Pooling'

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ['conv1']
        assert xp0_layers[1].bottoms == ['xinput0']
        assert xp0_layers[1].tops == ['pool1']
        assert xp0_layers[2].bottoms == ['conv1']
        assert xp0_layers[2].tops == []

    def test_two_partitions_through_interruption(self):
        # A layer inside a residual type branch os not supported
        # Here: BatchNorm
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
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv1'],
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 4, 3, 3],
                sizes=[36],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['pool1'],
                targets=[]
            ),
            XLayer(
                name='bn1',
                type=['BatchNorm'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=['concat1'],
                data=BatchData(np.array([1, 1]), np.array([0, 0]),
                               np.array([1, 1]), np.array([0, 0])),
                layer=['bn1'],
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
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv2'],
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert len(p_xgraph.get_layer_names()) == 6
        assert p_xgraph.get_subgraph_names() == ['xp0']

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ['Input']
        assert p_xlayers[1].type[0] in ['Convolution']
        assert p_xlayers[2].type[0] in ['Pooling']
        assert p_xlayers[3].type[0] in ['BatchNorm']
        assert p_xlayers[4].type[0] in ['Concat']
        assert p_xlayers[5].type[0] in ['Convolution']

        assert p_xlayers[0].target == 'cpu'
        assert p_xlayers[1].target == 'test'
        assert p_xlayers[2].target == 'test'
        assert p_xlayers[3].target == 'cpu'
        assert p_xlayers[4].target == 'cpu'
        assert p_xlayers[5].target == 'cpu'

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == 'xp0'
        assert p_xlayers[2].subgraph == 'xp0'
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph is None
        assert p_xlayers[5].subgraph is None

        assert p_xlayers[3].name == 'bn1'
        assert p_xlayers[3].bottoms == ['conv1']
        assert p_xlayers[3].tops == ['concat1']

        assert p_xlayers[4].name == 'concat1'
        assert p_xlayers[4].bottoms == ['pool1', 'bn1']
        assert p_xlayers[4].tops == ['conv2']

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == 'xp0'
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert xp0.bottoms == ['in1']
        assert xp0.tops == ['bn1', 'concat1']
        assert xp0.shapes == [[1, 2, 3, 3], [1, 4, 3, 3]]
        assert xp0.sizes == [18, 36]
        assert xp0.attrs['target'] == 'test'
        assert xp0.attrs['__bottom_tensors'] == {'xinput0': ['in1']}
        assert xp0.attrs['orig_bottom_tensors'] == {'xinput0': ['in1']}
        assert xp0.attrs['__top_tensors'] == \
            {'conv1': ['bn1'], 'pool1': ['concat1']}
        assert xp0.attrs['orig_top_tensors'] == \
            {'conv1': ['bn1'], 'pool1': ['concat1']}

        assert(len(xp0_xgraph) == 3)
        xp0_layers = xp0_xgraph.get_layers()

        assert [X.name for X in xp0_xgraph.get_input_layers()] == ['xinput0']
        # TODO: XGraph only recognizes output layers when they have no top
        #   layers
        assert [X.name for X in xp0_xgraph.get_output_layers()] ==\
            ['pool1']

        assert xp0_layers[0].type[0] == 'Input'
        assert xp0_layers[0].layer[0] == 'conv1'
        assert xp0_layers[1].type[0] == 'Convolution'
        assert xp0_layers[2].type[0] == 'Pooling'

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ['conv1']
        assert xp0_layers[1].bottoms == ['xinput0']
        assert xp0_layers[1].tops == ['pool1']
        assert xp0_layers[2].bottoms == ['conv1']
        assert xp0_layers[2].tops == []

    def test_multiple_partitions(self):
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
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv1'],
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
            ),
            XLayer(
                name='bn1',
                type=['BatchNorm'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['add1'],
                tops=['pool2'],
                data=BatchData(np.array([1, 1]), np.array([0, 0]),
                               np.array([1, 1]), np.array([0, 0])),
                layer=['bn1'],
                targets=[]
            ),
            XLayer(
                name='pool2',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['bn1'],
                tops=[],
                layer=['pool2'],
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert(len(p_xgraph.get_layer_names()) == 7)
        # ! Only xp0 because only one subgraph can exist for now (largest)
        assert(set(p_xgraph.get_subgraph_names()) == set(['xp0']))

        p_xlayers = p_xgraph.get_layers()
        assert(p_xlayers[0].type[0] in ['Input'])
        assert(p_xlayers[1].type[0] in ['Convolution'])
        assert(p_xlayers[2].type[0] in ['Pooling'])
        assert(p_xlayers[3].type[0] in ['Input'])
        assert(p_xlayers[4].type[0] in ['Eltwise'])
        assert(p_xlayers[5].type[0] in ['BatchNorm'])
        assert(p_xlayers[6].type[0] in ['Pooling'])

        assert(p_xlayers[0].target == 'cpu')
        assert(p_xlayers[1].target == 'test')
        assert(p_xlayers[2].target == 'test')
        assert(p_xlayers[3].target == 'cpu')
        assert(p_xlayers[4].target == 'cpu')
        assert(p_xlayers[5].target == 'cpu')
        # ! CPU because only one subgraph can exist for now (largest)
        assert(p_xlayers[6].target == 'cpu')

        assert(p_xlayers[0].subgraph is None)
        assert(p_xlayers[1].subgraph == 'xp0')
        assert(p_xlayers[2].subgraph == 'xp0')
        assert(p_xlayers[3].subgraph is None)
        assert(p_xlayers[4].subgraph is None)
        assert(p_xlayers[5].subgraph is None)
        assert(p_xlayers[6].subgraph is None)

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert(len(subgraphs) == 1)
        xp0 = subgraphs[0]
        assert(xp0.name == 'xp0')
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert(xp0.bottoms == ['in1'])
        assert(xp0.tops == ['add1'])
        assert(xp0.shapes == [[1, 2, 2, 2]])
        assert(xp0.sizes == [8])

        assert(len(xp0_xgraph) == 3)
        xp0_layers = xp0_xgraph.get_layers()

        assert(xp0_layers[0].type[0] == 'Input')
        assert(xp0_layers[0].layer[0] == 'conv1')
        assert(xp0_layers[1].type[0] == 'Convolution')
        assert(xp0_layers[2].type[0] == 'Pooling')

        assert(xp0_layers[0].bottoms == [])
        assert(xp0_layers[0].tops == ['conv1'])
        assert(xp0_layers[1].bottoms == ['xinput0'])
        assert(xp0_layers[1].tops == ['pool1'])
        assert(xp0_layers[2].bottoms == ['conv1'])
        assert(xp0_layers[2].tops == [])

    def test_multiple_partitions_largest_last(self):
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
                tops=['t1'],
                layer=['conv1'],
                data=ConvData(
                    weights=np.array([1, 1], dtype=np.float32),
                    biases=np.array([0, 0], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['conv1'],
                tops=['conv2'],
                layer=['t1'],
                targets=[],
                attrs={
                    'axes': [0, 2, 3, 1]
                }
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['t1'],
                tops=['pool1'],
                layer=['conv2'],
                data=ConvData(
                    weights=np.array([1, 1], dtype=np.float32),
                    biases=np.array([0, 0], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv2'],
                tops=[],
                layer=['pool1'],
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert len(p_xgraph.get_layer_names()) == 5
        # ! Only xp1 because only one subgraph can exist for now (largest)
        assert set(p_xgraph.get_subgraph_names()) == set(['xp1'])

        p_xlayers = p_xgraph.get_layers()
        assert(p_xlayers[0].type[0] in ['Input'])
        assert(p_xlayers[1].type[0] in ['Convolution'])
        assert(p_xlayers[2].type[0] in ['Transpose'])
        assert(p_xlayers[3].type[0] in ['Convolution'])
        assert(p_xlayers[4].type[0] in ['Pooling'])

        assert(p_xlayers[0].target == 'cpu')
        assert(p_xlayers[1].target == 'cpu')
        assert(p_xlayers[2].target == 'cpu')
        assert(p_xlayers[3].target == 'test')
        assert(p_xlayers[4].target == 'test')

        assert(p_xlayers[0].subgraph is None)
        assert(p_xlayers[1].subgraph is None)
        assert(p_xlayers[2].subgraph is None)
        assert(p_xlayers[3].subgraph == 'xp1')
        assert(p_xlayers[4].subgraph == 'xp1')

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert(len(subgraphs) == 1)
        xp1 = subgraphs[0]
        assert(xp1.name == 'xp1')
        xp1_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp1.subgraph_data)

        assert(xp1.bottoms == ['t1'])
        assert(xp1.tops == [])
        assert(xp1.shapes == [[1, 2, 2, 2]])
        assert(xp1.sizes == [8])

        assert(len(xp1_xgraph) == 3)
        xp1_layers = xp1_xgraph.get_layers()

        assert(xp1_layers[0].type[0] == 'Input')
        assert(xp1_layers[0].layer[0] == 'conv2')
        assert(xp1_layers[1].type[0] == 'Convolution')
        assert(xp1_layers[2].type[0] == 'Pooling')

        assert(xp1_layers[0].bottoms == [])
        assert(xp1_layers[0].tops == ['conv2'])
        assert(xp1_layers[1].bottoms == ['xinput0'])
        assert(xp1_layers[1].tops == ['pool1'])
        assert(xp1_layers[2].bottoms == ['conv2'])
        assert(xp1_layers[2].tops == [])

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
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv1'],
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
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['in2'],
                tops=['concat1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv2'],
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
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[],
                bottoms=['concat1'],
                tops=[],
                layer=['dense1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert len(p_xgraph.get_layer_names()) == 7
        assert p_xgraph.get_subgraph_names() == ['xp2']

        p_xlayers = p_xgraph.get_layers()

        assert p_xlayers[0].target == 'cpu'
        assert p_xlayers[1].target == 'test'
        assert p_xlayers[2].target == 'test'
        assert p_xlayers[3].target == 'cpu'
        assert p_xlayers[4].target == 'test'
        assert p_xlayers[5].target == 'test'
        assert p_xlayers[6].target == 'cpu'

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == 'xp2'
        assert p_xlayers[2].subgraph == 'xp2'
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph == 'xp2'
        assert p_xlayers[5].subgraph == 'xp2'
        assert p_xlayers[6].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert len(subgraphs) == 1
        xp2 = subgraphs[0]
        assert xp2.name == 'xp2'
        xp2_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp2.subgraph_data)

        assert xp2.bottoms == ['in1', 'in2']
        assert xp2.tops == ['dense1']
        assert xp2.shapes == [[1, 4, 2, 2]]
        assert xp2.sizes == [16]

        assert len(xp2_xgraph) == 6
        xp2_layers = xp2_xgraph.get_layers()

        assert(xp2_layers[0].type[0] == 'Input')
        assert(xp2_layers[0].layer[0] == 'conv1')
        assert(xp2_layers[1].type[0] == 'Convolution')
        assert(xp2_layers[2].type[0] == 'Pooling')
        assert(xp2_layers[3].type[0] == 'Input')
        assert(xp2_layers[3].layer[0] == 'conv2')
        assert(xp2_layers[4].type[0] == 'Convolution')
        assert(xp2_layers[5].type[0] == 'Concat')

        assert(xp2_layers[0].bottoms == [])
        assert(xp2_layers[0].tops == ['conv1'])
        assert(xp2_layers[1].bottoms == ['xinput0'])
        assert(xp2_layers[1].tops == ['pool1'])
        assert(xp2_layers[2].bottoms == ['conv1'])
        assert(xp2_layers[2].tops == ['concat1'])
        assert(xp2_layers[3].bottoms == [])
        assert(xp2_layers[3].tops == ['conv2'])
        assert(xp2_layers[4].bottoms == ['xinput1'])
        assert(xp2_layers[4].tops == ['concat1'])
        assert(xp2_layers[5].bottoms == ['pool1', 'conv2'])
        assert(xp2_layers[5].tops == [])

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
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 4, 3, 3],
                sizes=[],
                bottoms=['concat1'],
                tops=['pool1'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv1'],
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
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 4, 2, 2],
                sizes=[],
                bottoms=['concat1'],
                tops=['concat2'],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=['conv2'],
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
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)

        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert(len(p_xgraph.get_layer_names()) == 8)
        p_xlayers = p_xgraph.get_layers()

        assert(p_xlayers[0].target == 'cpu')
        assert(p_xlayers[1].target == 'cpu')
        assert(p_xlayers[2].target == 'test')
        assert(p_xlayers[3].target == 'test')
        assert(p_xlayers[4].target == 'test')
        assert(p_xlayers[5].target == 'test')
        assert(p_xlayers[6].target == 'test')
        assert(p_xlayers[7].target == 'cpu')

        assert(p_xlayers[0].subgraph is None)
        assert(p_xlayers[1].subgraph is None)
        assert(p_xlayers[2].subgraph == 'xp0')
        assert(p_xlayers[3].subgraph == 'xp0')
        assert(p_xlayers[4].subgraph == 'xp0')
        assert(p_xlayers[5].subgraph == 'xp0')
        assert(p_xlayers[6].subgraph == 'xp0')
        assert(p_xlayers[7].subgraph is None)

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert(len(subgraphs) == 1)
        xp0 = subgraphs[0]
        assert(xp0.name == 'xp0')
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert(xp0.bottoms == ['in1', 'in2'])
        assert(xp0.tops == ['dense1'])
        assert(xp0.shapes == [[1, 8, 2, 2]])
        assert(xp0.sizes == [32])

        assert(len(xp0_xgraph) == 7)
        xp0_layers = xp0_xgraph.get_layers()

        assert(xp0_layers[0].type[0] == 'Input')
        assert(xp0_layers[0].layer[0] == 'concat1')
        assert(xp0_layers[1].type[0] == 'Input')
        assert(xp0_layers[1].layer[0] == 'concat1')
        assert(xp0_layers[2].type[0] == 'Concat')
        assert(xp0_layers[3].type[0] == 'Convolution')
        assert(xp0_layers[4].type[0] == 'Pooling')
        assert(xp0_layers[5].type[0] == 'Convolution')
        assert(xp0_layers[6].type[0] == 'Concat')

        assert(xp0_layers[0].bottoms == [])
        assert(xp0_layers[0].tops == ['concat1'])
        assert(xp0_layers[1].bottoms == [])
        assert(xp0_layers[1].tops == ['concat1'])
        assert(xp0_layers[2].bottoms == ['xinput0', 'xinput1'])
        assert(xp0_layers[2].tops == ['conv1', 'conv2'])
        assert(xp0_layers[3].bottoms == ['concat1'])
        assert(xp0_layers[3].tops == ['pool1'])
        assert(xp0_layers[4].bottoms == ['conv1'])
        assert(xp0_layers[4].tops == ['concat2'])
        assert(xp0_layers[5].bottoms == ['concat1'])
        assert(xp0_layers[5].tops == ['concat2'])
        assert(xp0_layers[6].bottoms == ['pool1', 'conv2'])
        assert(xp0_layers[6].tops == [])

    def test_top_tensors_basic(self):
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
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1'],
                tops=['s1'],
                layer=['t1'],
                internal=1,
                targets=[],
                attrs={
                    'axes': [0, 2, 3, 1]
                }
            ),
            XLayer(
                name='s1',
                type=['Sqrt'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['t1'],
                tops=[],
                layer=['s1'],
                internal=0,
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert(len(p_xgraph.get_layer_names()) == 5)
        assert(p_xgraph.get_subgraph_names() == ['xp0'])

        p_xlayers = p_xgraph.get_layers()
        assert(p_xlayers[0].type[0] in ['Input'])
        assert(p_xlayers[1].type[0] in ['Convolution'])
        assert(p_xlayers[2].type[0] in ['Pooling'])
        assert(p_xlayers[3].type[0] in ['Transpose'])
        assert(p_xlayers[4].type[0] in ['Sqrt'])

        assert(p_xlayers[0].target == 'cpu')
        assert(p_xlayers[1].target == 'test')
        assert(p_xlayers[2].target == 'test')
        assert(p_xlayers[3].target == 'cpu')
        assert(p_xlayers[4].target == 'cpu')

        assert(p_xlayers[0].subgraph is None)
        assert(p_xlayers[1].subgraph == 'xp0')
        assert(p_xlayers[2].subgraph == 'xp0')
        assert(p_xlayers[3].subgraph is None)
        assert(p_xlayers[4].subgraph is None)

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == 'xp0'
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert xp0.bottoms == ['in1']
        assert xp0.tops == ['t1']
        assert xp0.shapes == [[1, 2, 2, 2]]
        assert xp0.sizes == [8]
        assert len(xp0_xgraph) == 3

        __bottom_tensors = xp0.attrs['__bottom_tensors']
        orig_bottom_tensors = xp0.attrs['orig_bottom_tensors']

        assert len(__bottom_tensors) == 1
        assert 'xinput0' in __bottom_tensors
        assert __bottom_tensors['xinput0'] == ['in1']

        assert len(orig_bottom_tensors) == 1
        assert 'xinput0' in orig_bottom_tensors
        assert orig_bottom_tensors['xinput0'] == ['in1']

        __top_tensors = xp0.attrs['__top_tensors']
        orig_top_tensors = xp0.attrs['orig_top_tensors']

        assert len(__top_tensors) == 1
        assert 'pool1' in __top_tensors
        assert __top_tensors['pool1'] == ['t1']

        assert len(orig_top_tensors) == 1
        assert 'pool1' in orig_top_tensors
        assert orig_top_tensors['pool1'] == ['s1']

    def test_multi_top_tensors(self):
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
                targets=[]
            ),
            XLayer(
                name='pool1',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv1'],
                tops=['t1', 't2'],
                layer=['pool1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1'],
                tops=['s1'],
                layer=['t1'],
                internal=1,
                targets=[],
                attrs={
                    'axes': [0, 2, 3, 1]
                }
            ),
            XLayer(
                name='t2',
                type=['Transpose'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1'],
                tops=['s2', 's3'],
                layer=['t2'],
                internal=1,
                targets=[],
                attrs={
                    'axes': [0, 2, 3, 1]
                }
            ),
            XLayer(
                name='s1',
                type=['Sqrt'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['t1'],
                tops=[],
                layer=['s1'],
                internal=0,
                targets=[]
            ),
            XLayer(
                name='s2',
                type=['Sqrt'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['t2'],
                tops=[],
                layer=['s2'],
                internal=0,
                targets=[]
            ),
            XLayer(
                name='s3',
                type=['Sqrt'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['t2'],
                tops=[],
                layer=['s3'],
                internal=0,
                targets=[]
            )
        ]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        TargetRegistry().annotate_ops(xgraph)
        p_xgraph = TestXGraphPartitioner.xgraph_partitioner.partition(
            xgraph, ['test']
        )

        assert len(p_xgraph.get_layer_names()) == 8
        assert p_xgraph.get_subgraph_names() == ['xp0']

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ['Input']
        assert p_xlayers[1].type[0] in ['Convolution']
        assert p_xlayers[2].type[0] in ['Pooling']
        assert p_xlayers[3].type[0] in ['Transpose']
        assert p_xlayers[4].type[0] in ['Sqrt']
        assert p_xlayers[5].type[0] in ['Transpose']
        assert p_xlayers[6].type[0] in ['Sqrt']
        assert p_xlayers[7].type[0] in ['Sqrt']

        assert p_xlayers[0].target == 'cpu'
        assert p_xlayers[1].target == 'test'
        assert p_xlayers[2].target == 'test'
        assert p_xlayers[3].target == 'cpu'
        assert p_xlayers[4].target == 'cpu'
        assert p_xlayers[5].target == 'cpu'
        assert p_xlayers[6].target == 'cpu'
        assert p_xlayers[7].target == 'cpu'

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == 'xp0'
        assert p_xlayers[2].subgraph == 'xp0'
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph is None
        assert p_xlayers[5].subgraph is None
        assert p_xlayers[6].subgraph is None
        assert p_xlayers[7].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(
            p_xgraph
        )

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == 'xp0'
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory\
            .build_from_xlayer(xp0.subgraph_data)

        assert xp0.bottoms == ['in1']
        assert xp0.tops == ['t1', 't2']
        assert xp0.shapes == [[1, 2, 2, 2], [1, 2, 2, 2]]
        assert xp0.sizes == [8, 8]
        assert len(xp0_xgraph) == 3

        __bottom_tensors = xp0.attrs['__bottom_tensors']
        orig_bottom_tensors = xp0.attrs['orig_bottom_tensors']

        assert len(__bottom_tensors) == 1
        assert 'xinput0' in __bottom_tensors
        assert __bottom_tensors['xinput0'] == ['in1']

        assert len(orig_bottom_tensors) == 1
        assert 'xinput0' in orig_bottom_tensors
        assert orig_bottom_tensors['xinput0'] == ['in1']

        __top_tensors = xp0.attrs['__top_tensors']
        orig_top_tensors = xp0.attrs['orig_top_tensors']

        assert len(__top_tensors) == 1
        assert 'pool1' in __top_tensors
        assert __top_tensors['pool1'] == ['t1', 't2']

        assert len(orig_top_tensors) == 1
        assert 'pool1' in orig_top_tensors
        assert orig_top_tensors['pool1'] == ['s1', 's2', 's3']
