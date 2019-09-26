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
Module for testing the xgraph base passing functionality


"""

import unittest

import numpy as np

from pyxir.graph.layer.xlayer import XLayer, ConvData, defaultXLayer
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.xgraph import XGraph
from pyxir.graph.passing.base_pass import XGraphBasePass

class TestPass(XGraphBasePass):

    def execute(self, xgraph):

        def replace_func(bottom_Xs, X, top_Xs):
            """ Replace Convolution with Pooling operation """

            new_Xs = []
            if X.type[0] in ['Convolution']:
                new_X = defaultXLayer()
                new_X = new_X._replace(
                    type = ['Pooling'],
                    name = X.name,
                    shapes = X.shapes,
                    sizes = X.sizes,
                    bottoms = X.bottoms,
                    tops = X.tops
                )
                new_Xs.append(new_X)
            else:
                new_Xs.append(X)

            return new_Xs

        new_xgraph = self._replace_layer_pass(
            xgraph = xgraph,
            replace_func = replace_func
        )

        return new_xgraph

                 

class TestXGraphBasePass(unittest.TestCase):

    xgraph_factory = XGraphFactory()

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
                tops=[],
                targets=[]
            )
        ]
        xgraph = TestXGraphBasePass.xgraph_factory.build_from_xlayer(xlayers)

        test_pass = TestPass()
        new_xgraph = test_pass.execute(xgraph)

        assert(len(new_xgraph) == 4)
        assert(new_xgraph.get('conv1').type[0] == 'Pooling')
