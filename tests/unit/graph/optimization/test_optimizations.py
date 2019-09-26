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
Module for testing XGraph optimizations


"""

import sys
import logging
import unittest
import numpy as np

from pyxir.graph.optimization import optimizations

from pyxir.graph.layer.xlayer import XLayer, ConvData, BatchData, ScaleData
from pyxir.graph.xgraph_factory import XGraphFactory

# logger = logging.getLogger('pyxir')
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.DEBUG)


class TestOptimizations(unittest.TestCase):

    xgraph_factory = XGraphFactory()

    def test_remove_simple(self):
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
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('conv1')
        bottom_Xs = xgraph.get_bottom_layers('conv1')
        top_Xs = xgraph.get_top_layers('conv1')

        optimizations.remove(xgraph, bottom_Xs, X, top_Xs)

        assert(len(xgraph) == 2)
        layers = xgraph.get_layers()
        assert(layers[0].type[0] == 'Input')
        assert(layers[1].type[0] == 'Pooling')

    def test_remove_complex(self):
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
                name='pool2',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv1'],
                tops=['add1'],
                layer=['pool2'],
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['pool1', 'pool2'],
                tops=[],
                layer=['add1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('conv1')
        bottom_Xs = xgraph.get_bottom_layers('conv1')
        top_Xs = xgraph.get_top_layers('conv1')

        optimizations.remove(xgraph, bottom_Xs, X, top_Xs)

        assert(len(xgraph) == 4)
        layers = xgraph.get_layers()
        assert(layers[0].type[0] == 'Input')
        assert(layers[1].type[0] == 'Pooling')
        assert(layers[2].type[0] == 'Pooling')
        assert(layers[3].type[0] == 'Eltwise')

    def test_merge_transposes_basic(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['t2'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='t2',
                type=['Transpose'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=['t1'],
                tops=['conv1'],
                layer=['t2'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['t2'],
                tops=[],
                layer=['conv1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('t1')
        bottom_Xs = xgraph.get_bottom_layers('t1')
        top_Xs = xgraph.get_top_layers('t1')

        optimizations.merge_transposes(xgraph, bottom_Xs, X, top_Xs)

        assert(len(xgraph) == 2)
        layers = xgraph.get_layers()

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Convolution')
        assert(layers[1].shapes == [1, 2, 3, 3])

    def test_merge_transposes_multiple_bottoms(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['concat1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t2'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='t2',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in2'],
                tops=['concat1'],
                layer=['t2'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 4, 4, 2],
                sizes=[32],
                bottoms=['t1', 't2'],
                tops=['t3', 't4'],
                layer=['concat1'],
                attrs={
                    'axis': 3
                },
                targets=[]
            ),
            XLayer(
                name='t3',
                type=['Transpose'],
                shapes=[1, 2, 4, 4],
                sizes=[32],
                bottoms=['concat1'],
                tops=[],
                layer=['t3'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='t4',
                type=['Transpose'],
                shapes=[1, 2, 4, 4],
                sizes=[32],
                bottoms=['concat1'],
                tops=[],
                layer=['t4'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('concat1')
        bottom_Xs = xgraph.get_bottom_layers('concat1')
        top_Xs = xgraph.get_top_layers('concat1')

        optimizations.merge_transposes(xgraph, bottom_Xs, X, top_Xs)

        assert(len(xgraph) == 7)
        layers = xgraph.get_layers()

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Transpose')
        assert(layers[1].name == 't1')
        assert(layers[1].shapes == [1, 4, 4, 1])
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['0_split_t3_t4'])

        assert(layers[2].type[0] == 'Transpose')
        assert(layers[2].name == '0_split_t3_t4')
        assert(layers[2].shapes == [1, 1, 4, 4])
        assert(layers[2].bottoms == ['t1'])
        assert(layers[2].tops == ['concat1'])

        assert(layers[3].type[0] == 'Input')
        assert(layers[3].shapes == [1, 1, 4, 4])

        assert(layers[4].type[0] == 'Transpose')
        assert(layers[4].name == 't2')
        assert(layers[4].shapes == [1, 4, 4, 1])
        assert(layers[4].bottoms == ['in2'])
        assert(layers[4].tops == ['1_split_t3_t4'])

        assert(layers[5].type[0] == 'Transpose')
        assert(layers[5].name == '1_split_t3_t4')
        assert(layers[5].shapes == [1, 1, 4, 4])
        assert(layers[5].bottoms == ['t2'])
        assert(layers[5].tops == ['concat1'])

        assert(layers[6].type[0] == 'Concat')
        assert(layers[6].name == 'concat1')
        assert(layers[6].shapes == [1, 2, 4, 4])
        assert(layers[6].bottoms == ['0_split_t3_t4', '1_split_t3_t4'])
        assert(layers[6].tops == [])

        X = xgraph.get('t1')
        bottom_Xs = xgraph.get_bottom_layers('t1')
        top_Xs = xgraph.get_top_layers('t1')

        optimizations.merge_transposes(xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert(len(xgraph) == 5)

        assert layers[0].type[0] == 'Input'
        assert layers[0].name == 'in1'
        assert layers[0].shapes == [1, 1, 4, 4]

        assert layers[1].type[0] == 'Input'
        assert layers[1].name == 'in2'
        assert layers[1].shapes == [1, 1, 4, 4]

        assert layers[2].type[0] == 'Transpose'
        assert layers[2].name == 't2'
        assert layers[2].shapes == [1, 4, 4, 1]
        assert layers[2].bottoms == ['in2']
        assert layers[2].tops == ['1_split_t3_t4']

        assert layers[3].type[0] == 'Transpose'
        assert layers[3].name == '1_split_t3_t4'
        assert layers[3].shapes == [1, 1, 4, 4]
        assert layers[3].bottoms == ['t2']
        assert layers[3].tops == ['concat1']

        assert layers[4].type[0] == 'Concat'
        assert layers[4].name == 'concat1'
        assert layers[4].shapes == [1, 2, 4, 4]
        assert layers[4].bottoms == ['in1', '1_split_t3_t4']
        assert layers[4].tops == []

    def test_merge_transpose_update_subgraph(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['pad1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='pad1',
                type=['Pad'],
                shapes=[1, 1, 6, 6],
                sizes=[36],
                bottoms=['in1'],
                tops=['t1'],
                layer=['pad1'],
                attrs={
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                },
                targets=[],
                subgraph='xp0'
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 6, 6, 1],
                sizes=[36],
                bottoms=['pad1'],
                tops=['conv1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 4, 4, 2],
                sizes=[32],
                bottoms=['t1'],
                tops=[],
                layer=['conv1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('pad1')
        bottom_Xs = xgraph.get_bottom_layers('pad1')
        top_Xs = xgraph.get_top_layers('pad1')

        optimizations.merge_transposes(xgraph, bottom_Xs, X, top_Xs)

        assert(len(xgraph) == 4)
        layers = xgraph.get_layers()

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Transpose')
        assert(layers[1].shapes == [1, 4, 4, 1])

        assert(layers[2].type[0] == 'Pad')
        assert(layers[2].shapes == [1, 6, 6, 1])

        assert(layers[3].type[0] == 'Convolution')
        assert(layers[3].shapes == [1, 4, 4, 2])

    def test_sweep_transposes_flow_basic(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['pad1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='pad1',
                type=['Pad'],
                shapes=[1, 6, 6, 1],
                sizes=[36],
                bottoms=['t1'],
                tops=['conv1'],
                layer=['pad1'],
                attrs={
                    'padding': [[0, 0], [1, 1], [1, 1], [0, 0]]
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['pad1'],
                tops=[],
                layer=['conv1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('pad1')
        bottom_Xs = xgraph.get_bottom_layers('pad1')
        top_Xs = xgraph.get_top_layers('pad1')

        optimizations.sweep_transposes_flow(
            xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert(len(xgraph) == 4)

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Pad')
        assert(layers[1].name == 'pad1')
        assert(layers[1].shapes == [1, 1, 6, 6])
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['moved_t1'])
        assert layers[1].attrs['padding'] ==\
            [[0, 0], [0, 0], [1, 1], [1, 1]]

        assert(layers[2].type[0] == 'Transpose')
        assert(layers[2].name == 'moved_t1')
        assert(layers[2].shapes == [1, 6, 6, 1])
        assert(layers[2].bottoms == ['pad1'])
        assert(layers[2].tops == ['conv1'])

        assert(layers[3].type[0] == 'Convolution')
        assert(layers[3].name == 'conv1')
        assert(layers[3].shapes == [1, 3, 3, 2])
        assert(layers[3].bottoms == ['moved_t1'])
        assert(layers[3].tops == [])

    def test_sweep_transposes_flow_basic_with_target(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                target='test',
                subgraph='xp0',
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['pad1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='pad1',
                type=['Pad'],
                shapes=[1, 6, 6, 1],
                sizes=[36],
                bottoms=['t1'],
                tops=['conv1'],
                layer=['pad1'],
                attrs={
                    'padding': [[0, 0], [1, 1], [1, 1], [0, 0]]
                },
                target='test',
                subgraph='xp0',
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['pad1'],
                tops=[],
                layer=['conv1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('pad1')
        bottom_Xs = xgraph.get_bottom_layers('pad1')
        top_Xs = xgraph.get_top_layers('pad1')

        optimizations.sweep_transposes_flow(
            xgraph, bottom_Xs, X, top_Xs, target='test')

        layers = xgraph.get_layers()
        assert(len(xgraph) == 4)

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Pad')
        assert(layers[1].name == 'pad1')
        assert(layers[1].shapes == [1, 1, 6, 6])
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['moved_t1'])
        assert layers[1].attrs['padding'] ==\
            [[0, 0], [0, 0], [1, 1], [1, 1]]

        assert(layers[2].type[0] == 'Transpose')
        assert(layers[2].name == 'moved_t1')
        assert(layers[2].shapes == [1, 6, 6, 1])
        assert(layers[2].bottoms == ['pad1'])
        assert(layers[2].tops == ['conv1'])

        assert(layers[3].type[0] == 'Convolution')
        assert(layers[3].name == 'conv1')
        assert(layers[3].shapes == [1, 3, 3, 2])
        assert(layers[3].bottoms == ['moved_t1'])
        assert(layers[3].tops == [])

    def test_sweep_transposes_flow_basic_with_target_no_change(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['pad1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='pad1',
                type=['Pad'],
                shapes=[1, 6, 6, 1],
                sizes=[36],
                bottoms=['t1'],
                tops=['conv1'],
                layer=['pad1'],
                attrs={
                    'padding': [[0, 0], [1, 1], [1, 1], [0, 0]]
                },
                target='test',
                subgraph='xp0',
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['pad1'],
                tops=[],
                layer=['conv1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('pad1')
        bottom_Xs = xgraph.get_bottom_layers('pad1')
        top_Xs = xgraph.get_top_layers('pad1')

        optimizations.sweep_transposes_flow(
            xgraph, bottom_Xs, X, top_Xs, target='test')

        layers = xgraph.get_layers()
        assert(len(xgraph) == 4)

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Transpose')
        assert(layers[1].name == 't1')
        assert(layers[1].shapes == [1, 4, 4, 1])
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['pad1'])

        assert(layers[2].type[0] == 'Pad')
        assert(layers[2].name == 'pad1')
        assert(layers[2].shapes == [1, 6, 6, 1])
        assert(layers[2].bottoms == ['t1'])
        assert(layers[2].tops == ['conv1'])
        assert layers[2].attrs['padding'] ==\
            [[0, 0], [1, 1], [1, 1], [0, 0]]

        assert(layers[3].type[0] == 'Convolution')
        assert(layers[3].name == 'conv1')
        assert(layers[3].shapes == [1, 3, 3, 2])
        assert(layers[3].bottoms == ['pad1'])
        assert(layers[3].tops == [])

    def test_sweep_transposes_flow_multiple_tops(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['add1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t2'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='t2',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in2'],
                tops=['add1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=[1, 4, 4, 2],
                sizes=[32],
                bottoms=['t1', 't2'],
                tops=['t3', 'add2'],
                layer=['add1'],
                attrs={},
                targets=[]
            ),
            XLayer(
                name='t3',
                type=['Transpose'],
                shapes=[1, 2, 4, 4],
                sizes=[32],
                bottoms=['add1'],
                tops=['conv1'],
                layer=['t3'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 2, 4, 4],
                sizes=[32],
                bottoms=['t3'],
                tops=['t4'],
                layer=['conv1'],
                attrs={'data_layout': 'NCHW'},
                targets=[]
            ),
            XLayer(
                name='t4',
                type=['Transpose'],
                shapes=[1, 4, 4, 2],
                sizes=[32],
                bottoms=['conv1'],
                tops=['add2'],
                layer=['t4'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='add2',
                type=['Eltwise'],
                shapes=[1, 4, 4, 4],
                sizes=[64],
                bottoms=['t4', 'add1'],
                tops=['dense1'],
                layer=['add2'],
                attrs={},
                targets=[]
            ),
            XLayer(
                name='dense1',
                type=['Dense'],
                shapes=[1, 20],
                sizes=[20],
                bottoms=['add2'],
                tops=[],
                layer=['dense1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('add1')
        bottom_Xs = xgraph.get_bottom_layers('add1')
        top_Xs = xgraph.get_top_layers('add1')

        optimizations.sweep_transposes_flow(xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert len(xgraph) == 10

        assert layers[0].name == 'in1'
        assert layers[0].tops == ['add1']

        assert layers[1].name == 'in2'
        assert layers[1].tops == ['add1']

        assert layers[2].type[0] == 'Eltwise'
        assert layers[2].name == 'add1'
        assert layers[2].shapes == [1, 2, 4, 4]

        assert layers[3].type[0] == 'Transpose'
        assert layers[3].shapes == [1, 4, 4, 2]
        assert layers[3].bottoms == ['add1']
        assert layers[3].tops == ['t3']

        assert layers[4].type[0] == 'Transpose'
        assert layers[4].shapes == [1, 2, 4, 4]
        assert layers[4].name == 't3'
        assert layers[4].tops == ['conv1']

        assert layers[5].type[0] == 'Convolution'
        assert layers[5].shapes == [1, 2, 4, 4]
        assert layers[5].tops == ['t4']

        assert layers[6].type[0] == 'Transpose'
        assert layers[6].name == 't4'
        assert layers[6].shapes == [1, 4, 4, 2]

        assert layers[7].type[0] == 'Transpose'
        assert layers[7].name == '1_split_t1_t2'
        assert layers[7].shapes == [1, 4, 4, 2]
        assert layers[7].bottoms == ['add1']
        assert layers[7].tops == ['add2']

        assert layers[8].type[0] == 'Eltwise'
        assert layers[8].name == 'add2'
        assert layers[8].bottoms == ['t4', '1_split_t1_t2']

        assert layers[9].type[0] == 'Dense'

        X = xgraph.get('t3')
        bottom_Xs = xgraph.get_bottom_layers('t3')
        top_Xs = xgraph.get_top_layers('t3')

        optimizations.sweep_transposes_flow(xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert len(xgraph) == 8

        assert layers[0].name == 'in1'
        assert layers[0].tops == ['add1']

        assert layers[1].name == 'in2'
        assert layers[1].tops == ['add1']

        assert layers[2].type[0] == 'Eltwise'
        assert layers[2].shapes == [1, 2, 4, 4]

        assert layers[3].type[0] == 'Convolution'
        assert layers[3].shapes == [1, 2, 4, 4]

        assert layers[4].type[0] == 'Transpose'
        assert layers[4].shapes == [1, 4, 4, 2]

        assert layers[5].type[0] == 'Transpose'
        assert layers[5].shapes == [1, 4, 4, 2]
        assert layers[5].bottoms == ['add1']
        assert layers[5].tops == ['add2']

        assert layers[6].type[0] == 'Eltwise'
        assert layers[6].name == 'add2'
        assert layers[6].shapes == [1, 4, 4, 4]

        X = xgraph.get('add2')
        bottom_Xs = xgraph.get_bottom_layers('add2')
        top_Xs = xgraph.get_top_layers('add2')

        optimizations.sweep_transposes_flow(xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert len(xgraph) == 7

        assert layers[2].type[0] == 'Eltwise'
        assert layers[2].shapes == [1, 2, 4, 4]
        assert layers[2].tops == ['conv1', 'add2']

        assert layers[3].type[0] == 'Convolution'
        assert layers[3].shapes == [1, 2, 4, 4]

        assert layers[4].type[0] == 'Eltwise'
        assert layers[4].shapes == [1, 4, 4, 4]

        assert layers[5].type[0] == 'Transpose'
        assert layers[5].shapes == [1, 4, 4, 4]
        assert layers[5].bottoms == ['add2']
        assert layers[5].tops == ['dense1']

    def test_sweep_transposes_flow_multiple_bottoms(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['conv1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['t1'],
                tops=['t2'],
                layer=['conv1'],
                attrs={
                    'data_layout': 'NHWC'
                },
                targets=[]
            ),
            XLayer(
                name='t2',
                type=['Transpose'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['t2'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=[],
                tops=['t3'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='t3',
                type=['Transpose'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['in2'],
                tops=['concat1'],
                layer=['t3'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 4, 3, 3],
                sizes=[32],
                bottoms=['t2', 't3'],
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
                sizes=[20],
                bottoms=['concat1'],
                tops=[],
                layer=['dense1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('concat1')
        bottom_Xs = xgraph.get_bottom_layers('concat1')
        top_Xs = xgraph.get_top_layers('concat1')

        optimizations.sweep_transposes_flow(xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert(len(xgraph) == 7)

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Transpose')
        assert(layers[1].name == 't1')
        assert(layers[1].shapes == [1, 4, 4, 1])
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['conv1'])

        assert(layers[2].type[0] == 'Convolution')
        assert(layers[2].name == 'conv1')
        assert(layers[2].shapes == [1, 3, 3, 2])
        assert(layers[2].bottoms == ['t1'])
        assert(layers[2].tops == ['concat1'])

        assert(layers[3].type[0] == 'Input')
        assert(layers[3].name == 'in2')
        assert(layers[3].shapes == [1, 3, 3, 2])

        assert(layers[4].type[0] == 'Concat')
        assert(layers[4].name == 'concat1')
        assert(layers[4].shapes == [1, 3, 3, 4])
        assert(layers[4].bottoms == ['conv1', 'in2'])
        assert(layers[4].tops == ['merge_t2_t3'])
        assert layers[4].attrs['axis'] == 3

        assert(layers[5].type[0] == 'Transpose')
        assert(layers[5].name == 'merge_t2_t3')
        assert(layers[5].shapes == [1, 4, 3, 3])
        assert(layers[5].bottoms == ['concat1'])
        assert(layers[5].tops == ['dense1'])

        assert(layers[6].type[0] == 'Dense')
        assert(layers[6].name == 'dense1')
        assert(layers[6].shapes == [1, 20])
        assert(layers[6].bottoms == ['merge_t2_t3'])
        assert(layers[6].tops == [])

        X = xgraph.get('dense1')
        bottom_Xs = xgraph.get_bottom_layers('dense1')
        top_Xs = xgraph.get_top_layers('dense1')

        optimizations.merge_transposes(xgraph, bottom_Xs, X, top_Xs)

        layers = xgraph.get_layers()
        assert(len(xgraph) == 7)

        assert(layers[0].type[0] == 'Input')
        assert(layers[0].shapes == [1, 1, 4, 4])

        assert(layers[1].type[0] == 'Transpose')
        assert(layers[1].name == 't1')
        assert(layers[1].shapes == [1, 4, 4, 1])
        assert(layers[1].bottoms == ['in1'])
        assert(layers[1].tops == ['conv1'])

        assert(layers[2].type[0] == 'Convolution')
        assert(layers[2].name == 'conv1')
        assert(layers[2].shapes == [1, 3, 3, 2])
        assert(layers[2].bottoms == ['t1'])
        assert(layers[2].tops == ['concat1'])

        assert(layers[3].type[0] == 'Input')
        assert(layers[3].name == 'in2')
        assert(layers[3].shapes == [1, 3, 3, 2])

        assert(layers[4].type[0] == 'Concat')
        assert(layers[4].name == 'concat1')
        assert(layers[4].shapes == [1, 3, 3, 4])
        assert(layers[4].bottoms == ['conv1', 'in2'])
        assert(layers[4].tops == ['merge_t2_t3'])
        assert layers[4].attrs['axis'] == 3

        assert(layers[5].type[0] == 'Transpose')
        assert(layers[5].name == 'merge_t2_t3')
        assert(layers[5].shapes == [1, 4, 3, 3])
        assert(layers[5].bottoms == ['concat1'])
        assert(layers[5].tops == ['dense1'])

        assert(layers[6].type[0] == 'Dense')
        assert(layers[6].name == 'dense1')
        assert(layers[6].shapes == [1, 20])
        assert(layers[6].bottoms == ['merge_t2_t3'])
        assert(layers[6].tops == [])

    def test_sweep_transposes_flow_multiple_bottoms_with_target(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['t1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='t1',
                type=['Transpose'],
                shapes=[1, 4, 4, 1],
                sizes=[16],
                bottoms=['in1'],
                tops=['conv1'],
                layer=['t1'],
                attrs={
                    'axes': [0, 2, 3, 1]
                },
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['t1'],
                tops=['t2'],
                layer=['conv1'],
                attrs={
                    'data_layout': 'NHWC'
                },
                target='test',
                subgraph='xp0',
                targets=[]
            ),
            XLayer(
                name='t2',
                type=['Transpose'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=['concat1'],
                layer=['t2'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=[1, 4, 4, 1],
                sizes=[18],
                bottoms=[],
                tops=['conv2'],
                layer=['in2'],
                targets=[]
            ),
            XLayer(
                name='conv2',
                type=['Convolution'],
                shapes=[1, 3, 3, 2],
                sizes=[18],
                bottoms=['in2'],
                tops=['t3'],
                layer=['conv2'],
                attrs={
                    'data_layout': 'NHWC'
                },
                target='test',
                subgraph='xp0',
                targets=[]
            ),
            XLayer(
                name='t3',
                type=['Transpose'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv2'],
                tops=['concat1', 'conv3'],
                layer=['t3'],
                attrs={
                    'axes': [0, 3, 1, 2]
                },
                targets=[]
            ),
            XLayer(
                name='concat1',
                type=['Concat'],
                shapes=[1, 4, 3, 3],
                sizes=[36],
                bottoms=['t2', 't3'],
                tops=['add1'],
                layer=['concat1'],
                attrs={
                    'axis': 1
                },
                target='test',
                subgraph='xp0',
                targets=[]
            ),
            XLayer(
                name='conv3',
                type=['Convolution'],
                shapes=[1, 4, 3, 3],
                sizes=[36],
                bottoms=['t3'],
                tops=['add1'],
                layer=['conv3'],
                attrs={
                    'data_layout': 'NCHW'
                },
                # Not in target
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=[1, 4, 3, 3],
                sizes=[20],
                bottoms=['concat1', 'conv3'],
                tops=[],
                layer=['add1'],
                targets=[]
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('concat1')
        bottom_Xs = xgraph.get_bottom_layers('concat1')
        top_Xs = xgraph.get_top_layers('concat1')

        optimizations.sweep_transposes_flow(xgraph, bottom_Xs, X, top_Xs,
                                            target='test')

        layers = xgraph.get_layers()
        assert len(xgraph) == 10

        assert layers[0].type[0] == 'Input'
        assert layers[0].shapes == [1, 1, 4, 4]

        assert layers[1].type[0] == 'Transpose'
        assert layers[1].name == 't1'
        assert layers[1].shapes == [1, 4, 4, 1]
        assert layers[1].bottoms == ['in1']
        assert layers[1].tops == ['conv1']

        assert layers[2].type[0] == 'Convolution'
        assert layers[2].name == 'conv1'
        assert layers[2].shapes == [1, 3, 3, 2]
        assert layers[2].bottoms == ['t1']
        assert layers[2].tops == ['concat1']
        assert layers[2].target == 'test'
        assert layers[2].subgraph == 'xp0'

        assert layers[3].type[0] == 'Input'
        assert layers[3].name == 'in2'
        assert layers[3].shapes == [1, 4, 4, 1]
        assert layers[3].tops == ['conv2']

        assert layers[4].type[0] == 'Convolution'
        assert layers[4].name == 'conv2'
        assert layers[4].shapes == [1, 3, 3, 2]
        assert layers[4].bottoms == ['in2']
        assert layers[4].tops == ['t3', 'concat1']
        assert layers[4].target == 'test'
        assert layers[4].subgraph == 'xp0'

        assert layers[5].type[0] == 'Concat'
        assert layers[5].name == 'concat1'
        assert layers[5].shapes == [1, 3, 3, 4]
        assert layers[5].bottoms == ['conv1', 'conv2']
        assert layers[5].tops == ['merge_t2_t3']
        assert layers[5].attrs['axis'] == 3
        assert layers[5].target == 'test'
        assert layers[5].subgraph == 'xp0'

        assert layers[6].type[0] == 'Transpose'
        assert layers[6].name == 'merge_t2_t3'
        assert layers[6].shapes == [1, 4, 3, 3]
        assert layers[6].bottoms == ['concat1']
        assert layers[6].tops == ['add1']
        assert layers[6].target == 'cpu'
        assert layers[6].subgraph is None

        assert layers[7].type[0] == 'Transpose'
        assert layers[7].name == 't3'
        assert layers[7].shapes == [1, 2, 3, 3]
        assert layers[7].bottoms == ['conv2']
        assert layers[7].tops == ['conv3']
        assert layers[7].target == 'cpu'
        assert layers[7].subgraph is None

        assert layers[8].type[0] == 'Convolution'
        assert layers[8].name == 'conv3'
        assert layers[8].shapes == [1, 4, 3, 3]
        assert layers[8].bottoms == ['t3']
        assert layers[8].tops == ['add1']
        assert layers[8].target == 'cpu'
        assert layers[8].subgraph is None

        assert layers[9].type[0] == 'Eltwise'
        assert layers[9].name == 'add1'
        assert layers[9].shapes == [1, 4, 3, 3]
        assert layers[9].bottoms == ['merge_t2_t3', 'conv3']
        assert layers[9].tops == []

    def test_merge_padding(self):
        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.])
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['pad1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='pad1',
                type=['Pad'],
                shapes=[1, 1, 6, 6],
                sizes=[36],
                bottoms=['in1'],
                tops=['conv1'],
                layer=['pad1'],
                targets=[],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [1, 1], [1, 1]]
                }
                # data=((0, 0), (0, 0), (1, 1), (1, 1))
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=[1, 2, 4, 4],
                sizes=[32],
                bottoms=['pad1'],
                tops=[],
                layer=['conv1'],
                data=ConvData(weights=W, biases=B),
                targets=[],
                attrs={
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]]
                }
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('pad1')
        bottom_Xs = xgraph.get_bottom_layers('pad1')
        top_Xs = xgraph.get_top_layers('pad1')

        optimizations.merge_padding(xgraph, bottom_Xs, X, top_Xs)

        assert len(xgraph) == 2
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'

        assert layers[1].type[0] == 'Convolution'
        assert layers[1].name == 'conv1'
        assert layers[1].bottoms == ['in1']
        assert layers[1].layer == ['pad1', 'conv1']
        assert layers[1].attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert layers[1].attrs['data_layout'] == 'NCHW'

    def test_merge_bias(self):
        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.])
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
                tops=['bias_add1'],
                layer=['conv1'],
                targets=[],
                attrs={
                    'relay_id': [1]
                },
                data=ConvData(weights=W, biases=B)
            ),
            XLayer(
                name='bias_add1',
                type=['Eltwise'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=[],
                layer=['bias_add1'],
                targets=[],
                attrs={
                    'relay_id': [2]
                },
                data=[np.array([1., 2.])]
            )
        ]
        np.testing.assert_array_equal(
            net[2].data[0], np.array([1., 2.]))
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('bias_add1')
        bottom_Xs = xgraph.get_bottom_layers('bias_add1')
        top_Xs = xgraph.get_top_layers('bias_add1')

        optimizations.merge_bias(xgraph, bottom_Xs, X, top_Xs)

        assert len(xgraph) == 2
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Convolution'
        assert layers[1].layer == ['conv1', 'bias_add1']
        np.testing.assert_array_equal(
            layers[1].data.biases, np.array([2., 1.]))

    def test_merge_conv_bn(self):
        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.])
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
                tops=['bn1'],
                layer=['conv1'],
                targets=[],
                attrs={},
                data=ConvData(weights=W, biases=B)
            ),
            XLayer(
                name='bn1',
                type=['BatchNorm'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=[],
                layer=['bn1'],
                targets=[],
                attrs={
                    'epsilon': 0.00000001
                },
                data=BatchData(mu=np.array([-1., 0.]),
                               sigma_square=np.array([4., 9.]),
                               gamma=np.array([1., 1.]),
                               beta=np.array([0., 0.]))
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('bn1')
        bottom_Xs = xgraph.get_bottom_layers('bn1')
        top_Xs = xgraph.get_top_layers('bn1')

        optimizations.merge_batchnorm_into_conv(xgraph, bottom_Xs, X, top_Xs)

        assert len(xgraph) == 2
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Convolution'
        assert layers[1].layer == ['conv1', 'bn1']

        weights_merged = np.reshape(
            np.array([[[1., 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2)) / np.array([2., 3.]).reshape((2, 1, 1, 1))
        np.testing.assert_array_almost_equal(
            layers[1].data.weights, weights_merged)
        np.testing.assert_array_almost_equal(
            layers[1].data.biases, np.array([1., -1./3.]))

    def test_merge_bn_scale(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 2, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['bn1'],
                layer=['in1'],
                targets=[]
            ),
            XLayer(
                name='bn1',
                type=['BatchNorm'],
                shapes=[1, 2, 4, 4],
                sizes=[16],
                bottoms=['in1'],
                tops=[],
                layer=['bn1'],
                targets=[],
                attrs={
                    'epsilon': 0.00000001,
                    'axis': 1
                },
                data=BatchData(mu=np.array([-1., 0.]),
                               sigma_square=np.array([4., 9.]),
                               gamma=np.array([1., 1.]),
                               beta=np.array([0., 0.]))
            ),
            XLayer(
                name='scale1',
                type=['Scale'],
                shapes=[1, 2, 4, 4],
                sizes=[16],
                bottoms=['bn1'],
                tops=[],
                layer=['scale1'],
                targets=[],
                attrs={'axis': 1},
                data=ScaleData(gamma=np.array([1., 2.]),
                               beta=np.array([-1., 1.]))
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('scale1')
        bottom_Xs = xgraph.get_bottom_layers('scale1')
        top_Xs = xgraph.get_top_layers('scale1')

        optimizations.merge_scale_into_conv_bn(xgraph, bottom_Xs, X, top_Xs)

        assert(len(xgraph) == 2)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'

        assert layers[1].type[0] == 'BatchNorm'
        assert layers[1].layer == ['bn1', 'scale1']

        gamma_merged = np.array([1., 2.])
        beta_merged = np.array([-1., 1.])
        np.testing.assert_array_almost_equal(
            layers[1].data.gamma, gamma_merged)
        np.testing.assert_array_almost_equal(
            layers[1].data.beta, beta_merged)

    def test_merge_conv_scale(self):
        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.])
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
                tops=['scale1'],
                layer=['conv1'],
                targets=[],
                attrs={},
                data=ConvData(weights=W, biases=B)
            ),
            XLayer(
                name='scale1',
                type=['Scale'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=['conv1'],
                tops=[],
                layer=['scale1'],
                targets=[],
                attrs={},
                data=ScaleData(gamma=np.array([1., 2.]),
                               beta=np.array([-1., 1.]))
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('scale1')
        bottom_Xs = xgraph.get_bottom_layers('scale1')
        top_Xs = xgraph.get_top_layers('scale1')

        optimizations.merge_scale_into_conv_bn(xgraph, bottom_Xs, X, top_Xs)

        assert len(xgraph) == 2
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'

        assert layers[1].type[0] == 'Convolution'
        assert layers[1].layer == ['conv1', 'scale1']

        weights_merged = np.reshape(
            np.array([[[1., 1], [0, 1]], [[6, 8], [-2, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        np.testing.assert_array_almost_equal(
            layers[1].data.weights, weights_merged)
        np.testing.assert_array_almost_equal(
            layers[1].data.biases, np.array([0., -1.]))

    def test_merge_relu(self):
        W = np.reshape(
            np.array([[[1, 1], [0, 1]], [[3, 4], [-1, 0]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([1., -1.])
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
                tops=['relu1'],
                layer=['conv1'],
                targets=[],
                attrs={},
                data=ConvData(weights=W, biases=B)
            ),
            XLayer(
                name='relu1',
                type=['ReLU'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['conv1'],
                tops=[],
                layer=['relu1'],
                targets=[],
                attrs={}
            )
        ]
        xgraph = TestOptimizations.xgraph_factory.build_from_xlayer(net)

        X = xgraph.get('relu1')
        bottom_Xs = xgraph.get_bottom_layers('relu1')
        top_Xs = xgraph.get_top_layers('relu1')

        optimizations.merge_relu(xgraph, bottom_Xs, X, top_Xs)

        assert len(xgraph) == 2
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'

        assert layers[1].type[0] == 'Convolution'
        assert layers[1].layer == ['conv1', 'relu1']
        assert layers[1].attrs['activation'] == 'ReLU'
