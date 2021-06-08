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

""" Module for testing the IO API's functionality """

import os
import unittest
import shutil
import numpy as np

import pyxir.io.api as api
from pyxir.shared.container import StrContainer, BytesContainer
from pyxir.opaque_func_registry import OpaqueFuncRegistry
from pyxir.shapes import TensorShape
from pyxir.graph.layer.xlayer import XLayer, ConvData, ScaleData, BatchData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.xgraph import XGraph


class TestIOAPIs(unittest.TestCase):

    xgraph_factory = XGraphFactory()

    def test_directory_serialization(self):
        dir_path = "/tmp/test_dir"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(dir_path)
        with open("/tmp/test_dir/test.txt", 'w') as f:
            f.write("testtest")

        bytes_c = BytesContainer(b"")
        of = OpaqueFuncRegistry.Get("pyxir.io.serialize_dir")
        of(dir_path, bytes_c)

        shutil.rmtree("/tmp/test_dir")
        assert not os.path.exists("/tmp/test_dir")

        of_de = OpaqueFuncRegistry.Get("pyxir.io.deserialize_dir")
        of_de(dir_path, bytes_c.get_bytes())

        assert os.path.exists("/tmp/test_dir")
        assert os.path.exists("/tmp/test_dir/test.txt")
        with open("/tmp/test_dir/test.txt", 'r') as f:
            assert f.read() == "testtest"

    def test_empty_directory_serialization(self):
        dir_path = "/tmp/test_dir"
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.makedirs(dir_path)

        bytes_c = BytesContainer(b"")
        of = OpaqueFuncRegistry.Get("pyxir.io.serialize_dir")
        of(dir_path, bytes_c)

        shutil.rmtree("/tmp/test_dir")
        assert not os.path.exists("/tmp/test_dir")

        of_de = OpaqueFuncRegistry.Get("pyxir.io.deserialize_dir")
        of_de(dir_path, bytes_c.get_bytes())

        assert os.path.exists("/tmp/test_dir")

    # def test_xgraph_serialization_params(self):
    #     net = [
    #         XLayer(
    #             name='in1',
    #             type=['Input'],
    #             shapes=TensorShape([1, 1, 4, 4]),
    #             bottoms=[],
    #             tops=[],
    #             targets=[]
    #         ),
    #         XLayer(
    #             name='in2',
    #             type=['Input'],
    #             shapes=TensorShape([1, 1, 4, 4]),
    #             bottoms=[],
    #             tops=[],
    #             targets=[]
    #         ),
    #         XLayer(
    #             name='add',
    #             type=['Eltwise'],
    #             shapes=TensorShape([1, 1, 4, 4]),
    #             bottoms=['in1', 'in2'],
    #             tops=[],
    #             targets=[]
    #         )
    #     ]
    #     xgraph = TestIOAPIs.xgraph_factory.build_from_xlayer(net)

    #     xgraph_str = api.get_xgraph_str(xgraph)

    #     xg = api.read_xgraph_str(xgraph_str)
    #     xg_layers = xg.get_layers()
    #     import pdb; pdb.set_trace()
    #     assert len(xgraph) == 3
        
    #     assert xg_layers[0].type[0] == 'Input'

    #     assert xg_layers[1].type[0] == 'Input'

    #     assert xg_layers[2].type[0] == 'Eltwise'

    def test_xgraph_serialization_basic(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=TensorShape([1, 1, 4, 4]),
                bottoms=[],
                tops=['add1'],
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=TensorShape([1, 2, 3, 3]),
                bottoms=[],
                tops=['add1'],
                targets=[]
            ),
            XLayer(
                name='conv1',
                type=['Convolution'],
                shapes=TensorShape([1, 2, 3, 3]),
                bottoms=['in1'],
                tops=['bias_add1'],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0., 1.], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='bias_add1',
                type=['BiasAdd'],
                shapes=TensorShape([1, 2, 3, 3]),
                bottoms=['conv1'],
                tops=['bn1'],
                data=[np.array([0., -1.], dtype=np.float32)],
                targets=[]
            ),
            XLayer(
                name='bn1',
                type=['BatchNorm'],
                shapes=TensorShape([1, 2, 3, 3]),
                bottoms=['bias_add1'],
                tops=['scale1'],
                data=BatchData(
                    mu=np.array([.5, 2.], dtype=np.float32),
                    sigma_square=np.array([1., 1.], dtype=np.float32),
                    gamma=np.array([.5, 2.], dtype=np.float32),
                    beta=np.array([0., -1.], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='scale1',
                type=['Scale'],
                shapes=TensorShape([1, 2, 3, 3]),
                bottoms=['bn1'],
                tops=['add1'],
                data=ScaleData(
                    np.array([.5, 2.], dtype=np.float32),
                    np.array([0., -1.], dtype=np.float32)
                ),
                targets=[]
            ),
            XLayer(
                name='add1',
                type=['Eltwise'],
                shapes=TensorShape([1, 2, 3, 3]),
                bottoms=['scale1', 'in2'],
                tops=[],
                targets=[]
            )
        ]
        xgraph = TestIOAPIs.xgraph_factory.build_from_xlayer(net)

        xgraph_str = api.get_xgraph_str(xgraph)

        xg = api.read_xgraph_str(xgraph_str)
        xg_layers = xg.get_layers()
        # import pdb; pdb.set_trace()
        
        assert len(xg_layers) == 7
        
        assert xg_layers[0].type[0] == 'Input'
        
        assert xg_layers[1].type[0] == 'Convolution'
        np.testing.assert_array_equal(
            xg_layers[1].data[0],
            np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
