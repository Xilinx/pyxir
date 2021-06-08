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

"""Module for testing the quantization simulation pass"""

import os
import sys
import logging
import unittest
import numpy as np

from pyxir import partition
from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.runtime.runtime_factory import RuntimeFactory
from pyxir.quantization.simulation.quant_sim_pass import XGraphQuantSimPass
from pyxir.target_registry import TargetRegistry, register_op_support_check

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Quantization Tensorflow related test because Tensorflow"
                            " is not available")

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

# logger = logging.getLogger('pyxir')
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(logging.DEBUG)


class TestQuantSimPass(unittest.TestCase):

    xgraph_factory = XGraphFactory()
    xf_exec_graph_factory = RuntimeFactory()

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

    def test_conv(self):
        W = np.reshape(
                np.array([[[1, 2], [3, 0]], [[1, 1], [0, 1]]],
                         dtype=np.float32),
                (2, 1, 2, 2))
        B = np.array([0., 0.], dtype=np.float32)

        net = [
            XLayer(
                name='in',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv2d0'],
                layer=['in'],
                targets=[]
            ),
            XLayer(
                name='conv2d0',
                type=['Convolution'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['in'],
                tops=[],
                layer=['conv2d0'],
                data=ConvData(W, B),
                attrs={
                    'data_layout': 'NCHW',
                    'kernel_layout': 'OIHW',
                    'kernel_size': [2, 2],
                    'shape': [1, 2, 3, 3],
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'dilation': [1, 1],
                    'groups': 1
                },
                targets=[]
            )
        ]
        xgraph = TestQuantSimPass.xgraph_factory.build_from_xlayer(
            net, name='test1'
        )

        quant_sim_pass = XGraphQuantSimPass(
            fdir=FILE_PATH,
            name=xgraph.get_name() + '_qsim'
        )
        qsim_xgraph = quant_sim_pass.execute(xgraph=xgraph,
                                             subgraphs_only=False)

        exec_graph = TestQuantSimPass.xf_exec_graph_factory.build_runtime(
            qsim_xgraph
        )

        inpts = {
            'in': np.reshape(
                np.array([
                    [10, 10, 0, 40],
                    [50, 10, 0, 80],
                    [30, 50, 10, 0],
                    [10, 90, 30, 40]],
                    dtype=np.float32
                ),
                (1, 1, 4, 4))
        }
        res = exec_graph.run(inpts)
        outpt = res[0]
        # for idx, layer, inpts, outpt, _ in exec_graph.run_stepwise(inpts):
        #     print(layer.name, outpt)

        expected_outpt = np.array([[[
            [182.28346,    36.45669,    80.20472],
            [160.40944,   160.40944,   189.5748],
            [160.40944,   342.6929,    102.078735]],

            [[29.165354,    7.2913384, 123.95275],
             [109.37008,   21.874016,   80.20472],
             [167.70079,    87.49606,    51.039368]]]],
            dtype=np.float32)

        np.testing.assert_array_almost_equal(outpt, expected_outpt, decimal=4)

    def test_conv_maxpool_subgraph(self):
        W = np.reshape(
            np.array([[[1, 2], [3, 0]], [[1, 1], [0, 1]]], dtype=np.float32),
            (2, 1, 2, 2))
        B = np.array([0., 0.], dtype=np.float32)

        net = [
            XLayer(
                name='in',
                type=['Input'],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=['conv2d0'],
                layer=['in'],
                targets=[]
            ),
            XLayer(
                name='conv2d0',
                type=['Convolution'],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=['in'],
                tops=[],
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
                name='max_pool2d0',
                type=['Pooling'],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=['conv2d0'],
                tops=[],
                layer=['max_pool2d0'],
                attrs={
                    'kernel_size': [2, 2],
                    'insize': [3, 3],
                    'outsize': [2, 2],
                    'data_layout': 'NCHW',
                    'padding': [[0, 0], [0, 0], [0, 0], [0, 0]],
                    'strides': [1, 1],
                    'pool_type': 'Max'
                },
                targets=[]
            )
        ]
        xgraph = TestQuantSimPass.xgraph_factory.build_from_xlayer(
            net, name='testtest'
        )
        p_xgraph = partition(xgraph, ['npu_test'])

        assert p_xgraph.get_layers()[0].target == 'cpu'
        assert p_xgraph.get_layers()[1].target == 'npu_test'
        assert p_xgraph.get_layers()[2].target == 'cpu'

        assert p_xgraph.get_layers()[0].subgraph is None
        assert p_xgraph.get_layers()[1].subgraph == 'xp0'
        assert p_xgraph.get_layers()[2].subgraph is None

        quant_sim_pass = XGraphQuantSimPass(
            fdir=FILE_PATH,
            name=xgraph.get_name() + '_qsim'
        )
        qsim_xgraph = quant_sim_pass.execute(xgraph=p_xgraph,
                                             subgraphs_only=True)

        exec_graph = TestQuantSimPass.xf_exec_graph_factory.build_runtime(
            qsim_xgraph
        )

        inpts = {
            'in': np.reshape(
                np.array([
                    [10, 10, 0, 40],
                    [50, 10, 0, 80],
                    [30, 50, 10, 0],
                    [10, 90, 30, 40]],
                    dtype=np.float32
                ),
                (1, 1, 4, 4))
        }
        res = exec_graph.run(inpts)
        outpt = res[0]

        expected_outpt = np.array([[
            [[182.28346, 189.5748],
             [342.6929, 342.6929]],
            [[109.37008, 123.95275],
             [167.70079, 87.49606]]]],
            dtype=np.float32)

        np.testing.assert_array_almost_equal(outpt, expected_outpt, decimal=4)
