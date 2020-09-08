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

""" Module for testing the xgraph IO functionality """

import os
import unittest

import numpy as np

from pyxir.shapes import TensorShape
from pyxir.graph.layer.xlayer import XLayer, ConvData, ScaleData, BatchData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.io.xgraph_io import XGraphIO


class TestXGraphIO(unittest.TestCase):

    xgraph_factory = XGraphFactory()
    xgraph_io = XGraphIO()

    def test_io_basic(self):
        net = [
            XLayer(
                name='in1',
                type=['Input'],
                shapes=TensorShape([1, 1, 4, 4]),
                bottoms=[],
                tops=[],
                targets=[]
            ),
            XLayer(
                name='in2',
                type=['Input'],
                shapes=TensorShape([1, 1, 4, 4]),
                bottoms=[],
                tops=[],
                targets=[]
            ),
            XLayer(
                name='add',
                type=['Eltwise'],
                shapes=TensorShape([1, 1, 4, 4]),
                bottoms=['in1', 'in2'],
                tops=[],
                targets=[]
            )
        ]
        xgraph = TestXGraphIO.xgraph_factory.build_from_xlayer(net)

        TestXGraphIO.xgraph_io.save(xgraph, 'test')

        loaded_xgraph = TestXGraphIO.xgraph_io.load('test.json', 'test.h5')

        # assert(len(loaded_xgraph.get_layers()) == len(xgraph.get_layers()))
        assert all([lxl == xl for lxl, xl in
                   zip(loaded_xgraph.get_layers(), xgraph.get_layers())])

        os.remove('test.json')
        os.remove('test.h5')

    def test_io_params(self):
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
        xgraph = TestXGraphIO.xgraph_factory.build_from_xlayer(net)
        xlayers = xgraph.get_layers()

        conv_weights = xlayers[1].data.weights
        conv_biases = xlayers[1].data.biases

        bias_add_biases = xlayers[2].data[0]

        bn_mu = xlayers[3].data.mu
        bn_var = xlayers[3].data.sigma_square
        bn_gamma = xlayers[3].data.gamma
        bn_beta = xlayers[3].data.beta

        scale_gamma = xlayers[4].data.gamma
        scale_beta = xlayers[4].data.beta

        TestXGraphIO.xgraph_io.save(xgraph, 'test')

        loaded_xgraph = TestXGraphIO.xgraph_io.load('test.json', 'test.h5')
        assert isinstance(loaded_xgraph.get_layers()[0].shapes, TensorShape)

        assert(len(loaded_xgraph.get_layers()) == len(xgraph.get_layers()))
        loaded_xlayers = loaded_xgraph.get_layers()

        loaded_conv_weights = loaded_xlayers[1].data.weights
        loaded_conv_biases = loaded_xlayers[1].data.biases
        loaded_bias_add_biases = loaded_xlayers[2].data[0]

        loaded_bn_mu = loaded_xlayers[3].data.mu
        loaded_bn_var = loaded_xlayers[3].data.sigma_square
        loaded_bn_gamma = loaded_xlayers[3].data.gamma
        loaded_bn_beta = loaded_xlayers[3].data.beta

        loaded_scale_gamma = loaded_xlayers[4].data.gamma
        loaded_scale_beta = loaded_xlayers[4].data.beta

        np.testing.assert_array_almost_equal(conv_weights, loaded_conv_weights)
        np.testing.assert_array_almost_equal(conv_biases, loaded_conv_biases)

        np.testing.assert_array_almost_equal(
            bias_add_biases, loaded_bias_add_biases)

        np.testing.assert_array_almost_equal(bn_mu, loaded_bn_mu)
        np.testing.assert_array_almost_equal(bn_var, loaded_bn_var)
        np.testing.assert_array_almost_equal(bn_gamma, loaded_bn_gamma)
        np.testing.assert_array_almost_equal(bn_beta, loaded_bn_beta)

        np.testing.assert_array_almost_equal(
            scale_gamma, loaded_scale_gamma)
        np.testing.assert_array_almost_equal(
            scale_beta, loaded_scale_beta)

        os.remove('test.json')
        os.remove('test.h5')


if __name__ == '__main__':
    unittest.main()
