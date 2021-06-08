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

import os
import onnx
import unittest
import numpy as np

from onnx import helper, AttributeProto, TensorProto, GraphProto

from pyxir.frontend.onnx.base import from_onnx, prequantize_onnx_model
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.graph import XGraph
from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.opaque_func_registry import OpaqueFuncRegistry
from pyxir.shared.quantizer_output import QuantizerOutput

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestONNXFrontend(unittest.TestCase):

    target_registry = TargetRegistry()

    @classmethod
    def setUpClass(cls):
        def xgraph_build_func(xgraph):
            raise NotImplementedError("")

        def xgraph_optimizer(xgraph, target):
            return xgraph

        def xgraph_quantizer(xgraph, inputs_func, **kwargs):
            # test_quant_file = os.path.join(FILE_DIR, 'test_quant_info.txt')
            # open(test_quant_file, 'w').close()
            # q_output = QuantizerOutput('xgraph')
            # q_output.add('xp0', None, test_quant_file, None)
            # xgraph.set_quantizer_output(q_output)
            for X in xgraph.get_layers():
                if 'Convolution' in X.type:
                    X.attrs['vai_quant'] = ['vai_quant_in', 'vai_quant_out',
                                            'vai_quant_weights',
                                            'vai_quant_biases']
                    X.attrs['vai_quant_in'] = [8, 8]
                    X.attrs['vai_quant_out'] = [8, 5]
                    X.attrs['vai_quant_weights'] = [5, 8]
                    X.attrs['vai_quant_biases'] = [5, 5]
                if 'Pooling' in X.type:
                    X.attrs['vai_quant'] = ['vai_quant_in', 'vai_quant_out']
                    X.attrs['vai_quant_in'] = [8, 8]
                    X.attrs['vai_quant_out'] = [8, 5]
            return xgraph

        def xgraph_compiler(xgraph):
            raise NotImplementedError("")

        cls.target_registry.register_target('test_dpu',
                                            xgraph_optimizer,
                                            xgraph_quantizer,
                                            xgraph_compiler,
                                            xgraph_build_func)

        @register_op_support_check('test_dpu', 'Convolution')
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test_dpu', 'BiasAdd')
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check('test_dpu', 'Pooling')
        def pooling_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestONNXFrontend.target_registry.unregister_target('test_dpu')

    def test_simple_model(self):
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                          [None, 1, 4, 4])
        x_val = np.array([[[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]]]]).astype(np.float32)
        # x_init = helper.make_tensor('x', TensorProto.FLOAT, (1, 1, 4, 4),
        #                             list(x_val.reshape(-1)))

        # Create one output (ValueInfoProto)
        z = helper.make_tensor_value_info('z', TensorProto.FLOAT,
                                          [None, 2, 2, 2])

        W_val = np.array([[[[1, 1],
                            [1, 1]]],
                          [[[1, -1],
                            [1, 1]]]]).astype(np.float32)
        W = helper.make_tensor('W', TensorProto.FLOAT, (2, 1, 2, 2),
                               list(W_val.reshape(-1)))

        B_val = np.array([1, -1]).astype(np.float32)
        B = helper.make_tensor('B', TensorProto.FLOAT, (2,),
                               list(B_val.reshape((-1))))

        conv_node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[1, 1, 0, 0]
        )

        pool_node = onnx.helper.make_node(
            'AveragePool',
            inputs=['y'],
            outputs=['z'],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2]
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            [conv_node, pool_node],
            'test-model',
            [x],
            [z],
            [W, B]  # x_init
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(graph_def,
                                           producer_name='onnx-example')

        xgraph = from_onnx(model_def)

        xlayers = xgraph.get_layers()
        assert len(xlayers) == 4

        assert xlayers[0].name == 'x'
        assert xlayers[0].type[0] == 'Input'
        assert xlayers[0].shapes == [-1, 1, 4, 4]
        assert xlayers[0].attrs['onnx_id'] == 'x'

        assert xlayers[1].name == 'y_Conv'
        assert xlayers[1].type[0] == 'Convolution'
        assert xlayers[1].shapes == [-1, 2, 4, 4]
        assert xlayers[1].attrs['padding'] == [(0, 0), (0, 0), (1, 0), (1, 0)]
        assert xlayers[1].attrs['strides'] == [1, 1]
        assert xlayers[1].attrs['dilation'] == [1, 1]
        assert xlayers[1].attrs['kernel_size'] == [2, 2]
        assert xlayers[1].attrs['channels'] == [1, 2]
        assert xlayers[1].attrs['data_layout'] == 'NCHW'
        assert xlayers[1].attrs['kernel_layout'] == 'OIHW'
        assert xlayers[1].attrs['groups'] == 1
        assert xlayers[1].attrs['onnx_id'] == 'y'

        assert xlayers[2].name == 'y'
        assert xlayers[2].shapes == [-1, 2, 4, 4]
        assert xlayers[2].attrs['axis'] == 1
        assert xlayers[2].attrs['onnx_id'] == 'y'

        assert xlayers[3].name == 'z'
        assert xlayers[3].shapes == [-1, 2, 2, 2]
        assert xlayers[3].type[0] == 'Pooling'
        assert xlayers[3].attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert xlayers[3].attrs['strides'] == [2, 2]
        assert xlayers[3].attrs['kernel_size'] == [2, 2]
        assert xlayers[3].attrs['data_layout'] == 'NCHW'
        assert xlayers[3].attrs['type'] == 'Avg'
        assert xlayers[3].attrs['onnx_id'] == 'z'

    def test_simple_model_opaque_func(self):
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                          [None, 1, 4, 4])
        x_val = np.array([[[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]]]]).astype(np.float32)
        # x_init = helper.make_tensor('x', TensorProto.FLOAT, (1, 1, 4, 4),
        #                             list(x_val.reshape(-1)))

        # Create one output (ValueInfoProto)
        z = helper.make_tensor_value_info('z', TensorProto.FLOAT,
                                          [None, 2, 2, 2])

        W_val = np.array([[[[1, 1],
                            [1, 1]]],
                          [[[1, -1],
                            [1, 1]]]]).astype(np.float32)
        W = helper.make_tensor('W', TensorProto.FLOAT, (2, 1, 2, 2),
                               list(W_val.reshape(-1)))

        B_val = np.array([1, -1]).astype(np.float32)
        B = helper.make_tensor('B', TensorProto.FLOAT, (2,),
                               list(B_val.reshape((-1))))

        conv_node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[1, 1, 0, 0]
        )

        pool_node = onnx.helper.make_node(
            'AveragePool',
            inputs=['y'],
            outputs=['z'],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2]
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            [conv_node, pool_node],
            'test-model',
            [x],
            [z],
            [W, B]  # x_init]
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(graph_def,
                                           producer_name='onnx-example')
        test_file = os.path.join(FILE_DIR, 'test.onnx')
        onnx.save(model_def, test_file)

        xgraph = XGraph(name='test')
        of = OpaqueFuncRegistry.Get('pyxir.onnx.from_onnx')
        of(xgraph, test_file)

        assert xgraph.get_name() == 'test-model'

        xlayers = xgraph.get_layers()
        assert len(xlayers) == 4

        assert xlayers[0].name == 'x'
        assert xlayers[0].type[0] == 'Input'
        assert xlayers[0].shapes == [-1, 1, 4, 4]
        assert xlayers[0].attrs['onnx_id'] == 'x'

        assert xlayers[1].name == 'y_Conv'
        assert xlayers[1].type[0] == 'Convolution'
        assert xlayers[1].shapes == [-1, 2, 4, 4]
        assert xlayers[1].attrs['padding'] == [(0, 0), (0, 0), (1, 0), (1, 0)]
        assert xlayers[1].attrs['strides'] == [1, 1]
        assert xlayers[1].attrs['dilation'] == [1, 1]
        assert xlayers[1].attrs['kernel_size'] == [2, 2]
        assert xlayers[1].attrs['channels'] == [1, 2]
        assert xlayers[1].attrs['data_layout'] == 'NCHW'
        assert xlayers[1].attrs['kernel_layout'] == 'OIHW'
        assert xlayers[1].attrs['groups'] == 1
        assert xlayers[1].attrs['onnx_id'] == 'y'

        assert xlayers[2].name == 'y'
        assert xlayers[2].shapes == [-1, 2, 4, 4]
        assert xlayers[2].attrs['axis'] == 1
        assert xlayers[2].attrs['onnx_id'] == 'y'

        assert xlayers[3].name == 'z'
        assert xlayers[3].shapes == [-1, 2, 2, 2]
        assert xlayers[3].type[0] == 'Pooling'
        assert xlayers[3].attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert xlayers[3].attrs['strides'] == [2, 2]
        assert xlayers[3].attrs['kernel_size'] == [2, 2]
        assert xlayers[3].attrs['data_layout'] == 'NCHW'
        assert xlayers[3].attrs['type'] == 'Avg'
        assert xlayers[3].attrs['onnx_id'] == 'z'

        of = OpaqueFuncRegistry.Get('pyxir.partition')
        of(xgraph, ['test_dpu'], "")

        assert xgraph.get_name() == 'test-model'
        assert len(xgraph) == 4

        xlayers = xgraph.get_layers()
        assert xlayers[0].name == 'x'
        assert xlayers[0].target == 'cpu'
        assert xlayers[0].subgraph is None

        assert xlayers[1].name == 'y_Conv'
        assert xlayers[1].target == 'test_dpu'
        assert xlayers[1].subgraph == 'xp0'

        assert xlayers[2].name == 'y'
        assert xlayers[2].type == ['BiasAdd']
        assert xlayers[2].target == 'test_dpu'
        assert xlayers[2].subgraph == 'xp0'

        assert xlayers[3].name == 'z'
        assert xlayers[3].type == ['Pooling']
        assert xlayers[3].target == 'test_dpu'
        assert xlayers[3].subgraph == 'xp0'

        os.remove(test_file)

    def test_prequantize_model(self):
        x = helper.make_tensor_value_info('x', TensorProto.FLOAT,
                                          [None, 1, 4, 4])
        x_val = np.array([[[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]]]]).astype(np.float32)
        # x_init = helper.make_tensor('x', TensorProto.FLOAT, (1, 1, 4, 4),
        #                             list(x_val.reshape(-1)))

        # Create one output (ValueInfoProto)
        z = helper.make_tensor_value_info('z', TensorProto.FLOAT,
                                          [None, 2, 2, 2])

        W_val = np.array([[[[1, 1],
                            [1, 1]]],
                          [[[1, -1],
                            [1, 1]]]]).astype(np.float32)
        W = helper.make_tensor('W', TensorProto.FLOAT, (2, 1, 2, 2),
                               list(W_val.reshape(-1)))

        B_val = np.array([1, -1]).astype(np.float32)
        B = helper.make_tensor('B', TensorProto.FLOAT, (2,),
                               list(B_val.reshape((-1))))

        conv_node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W', 'B'],
            outputs=['y'],
            kernel_shape=[2, 2],
            pads=[1, 0, 1, 0]
        )

        pool_node = onnx.helper.make_node(
            'AveragePool',
            inputs=['y'],
            outputs=['z'],
            kernel_shape=[2, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 2]
        )

        # Create the graph (GraphProto)
        graph_def = onnx.helper.make_graph(
            [conv_node, pool_node],
            'test-model',
            [x],
            [z],
            [W, B]  # x_init]
        )

        # Create the model (ModelProto)
        model_def = onnx.helper.make_model(graph_def,
                                           producer_name='onnx-example')

        test_file = os.path.join(FILE_DIR, 'test_pre.onnx')

        def inputs_func():
            pass

        prequantize_onnx_model(model_def, 'test_dpu', inputs_func,
                               test_file)

        new_onnx_model = onnx.load(test_file)

        new_xgraph = from_onnx(new_onnx_model)
        assert new_xgraph.get('y_Conv').attrs['vai_quant'] == \
            ['vai_quant_in', 'vai_quant_out', 'vai_quant_weights',
             'vai_quant_biases']
        assert new_xgraph.get('y_Conv').attrs['vai_quant_in'] == [8, 8]

        assert new_xgraph.get('z').attrs['vai_quant'] == \
            ['vai_quant_in', 'vai_quant_out']
        assert new_xgraph.get('z').attrs['vai_quant_in'] == [8, 8]

        quant_file = os.path.join(FILE_DIR, "quant_info.txt")
        new_xgraph.save_quant_info_txt(quant_file)

        os.remove(test_file)
        os.remove(quant_file)
