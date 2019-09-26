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
Module for testing the xgraph functionality


"""

import os
import unittest
import numpy as np

# ! Important for device registration
import pyxir

from pyxir.targets import qsim
from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.xgraph import XGraph

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestXGraph(unittest.TestCase):

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
        target_registry.unregister_target('qsim')

    def test_xgraph_meta_attrs(self):

        xgraph = XGraph("xg")
        assert len(xgraph.meta_attrs) == 0

        xgraph.meta_attrs["test_attr"] = "test_val"
        assert len(xgraph.meta_attrs) == 1
        assert xgraph.meta_attrs["test_attr"] == "test_val"

        xgraph.meta_attrs["test_attr2"] = {"test_key": "test_val"}
        assert len(xgraph.meta_attrs) == 2
        assert xgraph.meta_attrs["test_attr2"] == {"test_key": "test_val"}

        xgraph.meta_attrs = {
            "d_test_attr": ["t1", "t2"]
        }
        assert len(xgraph.meta_attrs) == 1
        assert xgraph.meta_attrs["d_test_attr"] == ["t1", "t2"]

    def test_xgraph_add_get(self):

        xgraph = XGraph()

        xgraph.add(XLayer(
            name='in1',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        assert len(xgraph) == 1
        assert len(xgraph.get_layer_names()) == 1
        assert len(xgraph.get_output_names()) == 1
        assert len(xgraph.get_input_names()) == 1

        assert isinstance(xgraph.get('in1'), XLayer)
        assert xgraph.get('in1').bottoms == []
        assert xgraph.get('in1').tops == []

        X_conv = XLayer(
            name='conv1',
            type=['Convolution'],
            bottoms=['in1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        )
        xgraph.add(X_conv)

        assert len(xgraph) == 2
        assert xgraph.get_layer_names() == ['in1', 'conv1']
        assert xgraph.get_output_names() == ['conv1']
        assert xgraph.get_input_names() == ['in1']

        assert xgraph.get('in1').tops == ['conv1']

        assert isinstance(xgraph.get('conv1'), XLayer)
        assert xgraph.get('conv1').bottoms == ['in1']
        assert xgraph.get('conv1').tops == []
        assert xgraph.get('conv1').type == ['Convolution']

        np.testing.assert_array_equal(
            xgraph.get('conv1').data.weights,
            np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            xgraph.get('conv1').data.biases,
            np.array([0., 1.], dtype=np.float32)
        )

        xgraph.get('conv1').data = ConvData(
            weights=xgraph.get('conv1').data.weights * 2,
            biases=xgraph.get('conv1').data.biases
        )
        np.testing.assert_array_equal(
            xgraph.get('conv1').data.weights,
            np.array([[[[2, 4], [6, 8]]]], dtype=np.float32)
        )

        xgraph.remove(X_conv.name)

        assert len(xgraph) == 1
        assert 'in1' in xgraph
        assert len(xgraph.get_layer_names()) == 1
        assert len(xgraph.get_output_names()) == 1
        assert len(xgraph.get_input_names()) == 1

    def test_xgraph_add_remove(self):

        xgraph = XGraph()
        xgraph.add(XLayer(
            name='in1',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        assert(len(xgraph) == 1)
        assert(len(xgraph.get_layer_names()) == 1)
        assert(len(xgraph.get_output_names()) == 1)
        assert(len(xgraph.get_input_names()) == 1)

        X_conv = XLayer(
            name='conv1',
            type=['Convolution'],
            bottoms=['in1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        )
        xgraph.add(X_conv)

        assert(len(xgraph) == 2)
        assert(len(xgraph.get_layer_names()) == 2)
        assert(len(xgraph.get_output_names()) == 1)
        assert(len(xgraph.get_input_names()) == 1)

        xgraph.remove(X_conv.name)

        assert(len(xgraph) == 1)
        assert(len(xgraph.get_layer_names()) == 1)
        assert(len(xgraph.get_output_names()) == 1)
        assert(len(xgraph.get_input_names()) == 1)

    def test_xgraph_insert(self):

        xgraph = XGraph()
        xgraph.add(XLayer(
            name='in1',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        assert(len(xgraph) == 1)
        assert(len(xgraph.get_layer_names()) == 1)
        assert(len(xgraph.get_output_names()) == 1)
        assert(len(xgraph.get_input_names()) == 1)

        X_conv = XLayer(
            name='conv1',
            type=['Convolution'],
            bottoms=['in1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        )
        xgraph.add(X_conv)

        assert len(xgraph) == 2
        assert len(xgraph.get_layer_names()) == 2
        assert len(xgraph.get_output_names()) == 1
        assert len(xgraph.get_input_names()) == 1

        X_pool = XLayer(
            name='pool1',
            type=['Pooling'],
            bottoms=['in1'],
            tops=['conv1'],
            targets=[]
        )
        xgraph.insert(X_pool)

        assert len(xgraph) == 3
        assert len(xgraph.get_layer_names()) == 3
        assert len(xgraph.get_output_names()) == 1
        assert len(xgraph.get_input_names()) == 1

        xlayers = xgraph.get_layers()
        assert xlayers[0].name == 'in1'
        assert xlayers[0].bottoms == []
        assert xlayers[0].tops == ['pool1']
        assert xlayers[1].name == 'pool1'
        assert xlayers[1].bottoms == ['in1']
        assert xlayers[1].tops == ['conv1']
        assert xlayers[2].name == 'conv1'
        assert xlayers[2].bottoms == ['pool1']
        assert xlayers[2].tops == []

        X_in2 = XLayer(
            name='in2',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        )
        xgraph.add(X_in2)

        X_add = XLayer(
            name='add1',
            type=['Eltwise'],
            bottoms=['conv1', 'in2'],
            tops=[],
            targets=[]
        )
        xgraph.add(X_add)

        X_conv2 = XLayer(
            name='conv2',
            type=['Convolution'],
            bottoms=['in2'],
            tops=['add1'],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        )
        xgraph.insert(X_conv2)

        assert(len(xgraph) == 6)
        assert(len(xgraph.get_layer_names()) == 6)
        assert(len(xgraph.get_output_names()) == 1)
        assert(len(xgraph.get_input_names()) == 2)

        xlayers = xgraph.get_layers()
        assert xlayers[0].name == 'in1'
        assert xlayers[0].bottoms == []
        assert xlayers[0].tops == ['pool1']
        assert xlayers[1].name == 'pool1'
        assert xlayers[1].bottoms == ['in1']
        assert xlayers[1].tops == ['conv1']
        assert xlayers[2].name == 'conv1'
        assert xlayers[2].bottoms == ['pool1']
        assert xlayers[2].tops == ['add1']
        assert xlayers[3].name == 'in2'
        assert xlayers[3].bottoms == []
        assert xlayers[3].tops == ['conv2']
        assert xlayers[4].name == 'conv2'
        assert xlayers[4].bottoms == ['in2']
        assert xlayers[4].tops == ['add1']
        assert xlayers[5].name == 'add1'
        assert xlayers[5].bottoms == ['conv1', 'conv2']
        assert xlayers[5].tops == []

    def test_xgraph_device_tagging(self):

        xgraph = XGraph()
        xgraph.add(XLayer(
            name='in1',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='in2',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='conv1',
            type=['Convolution'],
            bottoms=['in1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='add1',
            type=['Eltwise'],
            bottoms=['conv1', 'in2'],
            tops=[],
            targets=[]
        ))

        xgraph.insert(XLayer(
            name='conv2',
            type=['Convolution'],
            bottoms=['in2'],
            tops=['add1'],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='pool1',
            type=['Pooling'],
            bottoms=['add1'],
            tops=[],
            targets=[]
        ))

        assert len(xgraph) == 6
        xlayers = xgraph.get_layers()
        assert xgraph.get_layer_names() == \
            ['in1', 'conv1', 'in2', 'conv2', 'add1', 'pool1']
        assert set(xlayers[0].targets) == set(['cpu', 'qsim'])
        assert set(xlayers[1].targets) == set(['cpu', 'qsim', 'test'])
        assert set(xlayers[2].targets) == set(['cpu', 'qsim'])
        assert set(xlayers[3].targets) == set(['cpu', 'qsim', 'test'])
        assert set(xlayers[4].targets) == set(['cpu', 'qsim'])
        assert set(xlayers[5].targets) == set(['cpu', 'qsim', 'test'])

        xgraph.remove('conv1')
        assert len(xgraph) == 5
        xlayers = xgraph.get_layers()

        assert xgraph.get_layer_names() == \
            ['in1', 'in2', 'conv2', 'add1', 'pool1']

        assert xlayers[3].type[0] == 'Eltwise'
        assert xlayers[3].bottoms == ['in1', 'conv2']

        assert set(xlayers[0].targets) == set(['cpu', 'qsim'])
        assert set(xlayers[1].targets) == set(['cpu', 'qsim'])
        assert set(xlayers[2].targets) == set(['cpu', 'qsim', 'test'])
        assert set(xlayers[3].targets) == set(['cpu', 'qsim'])
        assert set(xlayers[4].targets) == set(['cpu', 'qsim', 'test'])

    def test_copy(self):

        xgraph = XGraph()
        xgraph.add(XLayer(
            name='in1',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='in2',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='conv1',
            type=['Convolution'],
            bottoms=['in1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='add1',
            type=['Eltwise'],
            bottoms=['conv1', 'in2'],
            tops=[],
            targets=[]
        ))

        xgraph.insert(XLayer(
            name='conv2',
            type=['Convolution'],
            bottoms=['in2'],
            tops=['add1'],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='pool1',
            type=['Pooling'],
            bottoms=['add1'],
            tops=[],
            targets=[]
        ))

        assert len(xgraph) == 6
        assert xgraph.get_layer_names() == \
            ['in1', 'conv1', 'in2', 'conv2', 'add1', 'pool1']

        xg_copy = xgraph.copy()
        assert len(xg_copy) == 6
        assert xg_copy.get_layer_names() == \
            ['in1', 'conv1', 'in2', 'conv2', 'add1', 'pool1']
        xgc_layers = xg_copy.get_layers()

        assert xgc_layers[1].type == ['Convolution']
        assert xg_copy.get('conv1').type == ['Convolution']

        xgc_layers[1].type = ['Convolution2']
        assert xg_copy.get('conv1').type == ['Convolution2']

        xgc_layers[1].type = ['Convolution']
        assert xgc_layers[1].type == ['Convolution']
        assert xg_copy.get('conv1').type == ['Convolution']

        np.testing.assert_array_equal(
            xgc_layers[1].data.weights,
            np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            xgc_layers[1].data.biases,
            np.array([0., 1.], dtype=np.float32)
        )

        xgraph.get('conv1').data = ConvData(
            weights=xgc_layers[1].data.weights * 2,
            biases=xgc_layers[1].data.biases
        )

        np.testing.assert_array_equal(
            xgraph.get('conv1').data.weights,
            np.array([[[[2, 4], [6, 8]]]], dtype=np.float32)
        )

        np.testing.assert_array_equal(
            xgc_layers[1].data.weights,
            np.array([[[[1, 2], [3, 4]]]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            xgc_layers[1].data.biases,
            np.array([0., 1.], dtype=np.float32)
        )

    def test_visualize(self):

        xgraph = XGraph()
        xgraph.add(XLayer(
            name='in1',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='in2',
            type=['Input'],
            bottoms=[],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='conv1',
            type=['Convolution'],
            bottoms=['in1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='add1',
            type=['Eltwise'],
            bottoms=['conv1', 'in2'],
            tops=[],
            targets=[]
        ))

        xgraph.insert(XLayer(
            name='conv2',
            type=['Convolution'],
            bottoms=['in2'],
            tops=['add1'],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='conv3',
            type=['Convolution'],
            bottoms=['add1'],
            tops=[],
            data=ConvData(
                weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                biases=np.array([0., 1.], dtype=np.float32)
            ),
            targets=[]
        ))

        xgraph.add(XLayer(
            name='pool1',
            type=['Pooling'],
            bottoms=['add1'],
            tops=[],
            targets=[]
        ))

        xgraph.add(XLayer(
            name='add2',
            type=['Eltwise'],
            bottoms=['conv3', 'pool1'],
            tops=[],
            targets=[]
        ))

        assert len(xgraph) == 8
        assert xgraph.get_layer_names() == \
            ['in1', 'conv1', 'in2', 'conv2', 'add1', 'conv3', 'pool1', 'add2']

        out_file = os.path.join(FILE_DIR, 'viz.png')
        xgraph.visualize(out_file)

        os.remove(out_file)
