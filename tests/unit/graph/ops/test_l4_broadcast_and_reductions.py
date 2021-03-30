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
Module for testing the XOp factory and property functionality


"""

import unittest
import numpy as np
import pyxir as px

from pyxir.shapes import TensorShape, TupleShape
from pyxir.graph.layer.xlayer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph import ops


class TestL4BroadcastAndReductions(unittest.TestCase):
    def test_mean_layer_basic(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        # sX = xlf.get_xop_factory_func('Mean')(
        #     'mean', iX, [2, 3], True, False)
        sX = px.ops.mean("mean", iX, axes=[2, 3], keepdims=True, exclude=False)

        assert sX.type[0] == "Mean"
        assert sX.shapes.tolist() == [1, 2, 1, 1]
        assert sX.sizes == [2]
        assert sX.attrs["axes"] == [2, 3]
        assert sX.attrs["keepdims"] is True
        assert sX.bottoms == ["in1"]

        from pyxir.graph.ops.l4_broadcast_and_reductions import mean_transpose_transform

        mean_transpose_transform(sX, (0, 2, 3, 1))

        assert sX.type[0] == "Mean"
        assert sX.shapes == [1, 1, 1, 2]
        assert sX.attrs["axes"] == [1, 2]

    def test_mean_layer_keepdims(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        # sX = xlf.get_xop_factory_func('Mean')(
        #     'mean', [2, 3], False, False, iX)
        sX = px.ops.mean("mean", iX, axes=[2, 3], keepdims=False, exclude=False)

        assert sX.type[0] == "Mean"
        assert sX.shapes.tolist() == [1, 2]
        assert sX.sizes == [2]
        assert sX.attrs["axes"] == [2, 3]
        assert sX.attrs["keepdims"] is False
        assert sX.bottoms == ["in1"]

    def test_mean_layer_exclude(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        # sX = xlf.get_xop_factory_func('Mean')(
        #     'mean', [0, 1], False, True, iX)
        sX = px.ops.mean("mean", iX, axes=[0, 1], keepdims=False, exclude=True)

        assert sX.type[0] == "Mean"
        assert sX.shapes.tolist() == [1, 2]
        assert sX.sizes == [2]
        assert sX.attrs["axes"] == [2, 3]
        assert sX.attrs["keepdims"] is False
        assert sX.bottoms == ["in1"]
