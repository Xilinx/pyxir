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
"""Module for testing the AddExplicitOutputLayers transformation pass"""

import unittest
import numpy as np
import pyxir as px

from typing import List

from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.transformers.add_explicit_output_layers import AddExplicitOutputLayers


class TestAddExplicitOutputLayers(unittest.TestCase):

    xgraph_factory = XGraphFactory()

    def test_one_hidden_output_layer(self):
        shape = [-1, 4, 4, 4]
        x1 = px.ops.input("in1", shape=shape)
        x2 = px.ops.input("in2", shape=shape)
        concat = px.ops.concat("concat", [x1, x2], axis=3)
        k = px.ops.constant("k", np.ones((16, 8, 2, 2), dtype=np.float32))
        conv = px.ops.conv2d("conv", concat, k, [2, 2], data_layout="NHWC")

        xgraph = TestAddExplicitOutputLayers.xgraph_factory.build_from_xlayer(
            [x1, x2, concat, conv]
        )

        assert len(xgraph) == 4
        assert xgraph.get_output_names() == ["conv"]

        xgraph = AddExplicitOutputLayers(["concat"], layout="NHWC")(xgraph)

        layers = xgraph.get_layers()
        assert len(xgraph) == 5
        assert len(layers) == 5

        assert layers[0].type[0] == "Input"
        assert layers[0].name == "in1"
        assert layers[0].tops == ["concat_hidden"]
        assert layers[0].bottoms == []

        assert layers[1].type[0] == "Input"
        assert layers[1].name == "in2"
        assert layers[1].tops == ["concat_hidden"]
        assert layers[1].bottoms == []

        assert layers[2].type[0] == "Concat"
        assert layers[2].name == "concat_hidden"
        assert layers[2].tops == ["concat", "conv"]
        assert layers[2].bottoms == ["in1", "in2"]

        assert layers[3].type[0] == "Convolution"
        assert layers[3].name == "concat"
        assert layers[3].tops == []
        assert layers[3].bottoms == ["concat_hidden"]
        assert layers[3].shapes == [-1, 4, 4, 8]

        assert layers[4].type[0] == "Convolution"
        assert layers[4].name == "conv"
        assert layers[4].tops == []
        assert layers[4].bottoms == ["concat_hidden"]

    def test_multiple_hidden_output_layers(self):
        shape = [-1, 4, 4, 2]
        x1 = px.ops.input("in1", shape=shape)
        x2 = px.ops.input("in2", shape=shape)
        concat = px.ops.concat("concat", [x1, x2], axis=3)
        k = px.ops.constant("k", np.ones((1, 2, 2, 4), dtype=np.float32))
        conv = px.ops.conv2d("conv", concat, k, [2, 2])
        pool = px.ops.pool2d("pool", conv, "Avg", [2, 2])

        xgraph = TestAddExplicitOutputLayers.xgraph_factory.build_from_xlayer(
            [x1, x2, concat, conv, pool]
        )

        assert len(xgraph) == 5
        assert xgraph.get_output_names() == ["pool"]

        xgraph = AddExplicitOutputLayers(["concat", "conv"], layout="NHWC")(xgraph)

        layers = xgraph.get_layers()
        assert len(xgraph) == 7
        assert len(layers) == 7

        assert layers[0].type[0] == "Input"
        assert layers[0].name == "in1"
        assert layers[0].tops == ["concat_hidden"]
        assert layers[0].bottoms == []

        assert layers[1].type[0] == "Input"
        assert layers[1].name == "in2"
        assert layers[1].tops == ["concat_hidden"]
        assert layers[1].bottoms == []

        assert layers[2].type[0] == "Concat"
        assert layers[2].name == "concat_hidden"
        assert layers[2].tops == ["concat", "conv_hidden"]
        assert layers[2].bottoms == ["in1", "in2"]

        assert layers[3].type[0] == "Convolution"
        assert layers[3].name == "concat"
        assert layers[3].tops == []
        assert layers[3].bottoms == ["concat_hidden"]

        assert layers[4].type[0] == "Convolution"
        assert layers[4].name == "conv_hidden"
        assert layers[4].tops == ["conv", "pool"]
        assert layers[4].bottoms == ["concat_hidden"]

        assert layers[5].type[0] == "Convolution"
        assert layers[5].name == "conv"
        assert layers[5].tops == []
        assert layers[5].bottoms == ["conv_hidden"]

        assert layers[6].type[0] == "Pooling"
        assert layers[6].name == "pool"
        assert layers[6].tops == []
        assert layers[6].bottoms == ["conv_hidden"]
