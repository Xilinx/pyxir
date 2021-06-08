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

"""Module for testing the XOp factory and property functionality"""

import unittest
import numpy as np
import pyxir as px

from pyxir.graph.layer.xlayer import XLayer
from pyxir.graph import ops
from pyxir.graph.layer import xlayer_factory as xlf


class TestL1BasicNN(unittest.TestCase):
    def test_expand_dims_positive_axis(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[-1, 4],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        edX = xlf.get_xop_factory_func("ExpandDims")("ed1", [iX], axis=0, num_newaxis=2)

        assert edX.type[0] == "ExpandDims"
        assert edX.attrs["axis"] == 0
        assert edX.attrs["num_newaxis"] == 2
        assert edX.shapes == [1, 1, -1, 4]

        edX = xlf.get_xop_factory_func("ExpandDims")("ed2", [iX], axis=1, num_newaxis=2)

        assert edX.type[0] == "ExpandDims"
        assert edX.attrs["axis"] == 1
        assert edX.attrs["num_newaxis"] == 2
        assert edX.shapes == [-1, 1, 1, 4]

        edX = xlf.get_xop_factory_func("ExpandDims")("ed3", [iX], axis=2, num_newaxis=2)

        assert edX.type[0] == "ExpandDims"
        assert edX.attrs["axis"] == 2
        assert edX.attrs["num_newaxis"] == 2
        assert edX.shapes == [-1, 4, 1, 1]

        with self.assertRaises(AssertionError):
            edX = xlf.get_xop_factory_func("ExpandDims")(
                "ed4", [iX], axis=3, num_newaxis=2
            )

    def test_expand_dims_negative_axis(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[-1, 4],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        edX = xlf.get_xop_factory_func("ExpandDims")(
            "ed1", [iX], axis=-1, num_newaxis=2
        )

        assert edX.type[0] == "ExpandDims"
        assert edX.attrs["axis"] == -1
        assert edX.attrs["num_newaxis"] == 2
        assert edX.shapes == [-1, 4, 1, 1]

        edX = xlf.get_xop_factory_func("ExpandDims")(
            "ed2", [iX], axis=-2, num_newaxis=2
        )

        assert edX.type[0] == "ExpandDims"
        assert edX.attrs["axis"] == -2
        assert edX.attrs["num_newaxis"] == 2
        assert edX.shapes == [-1, 1, 1, 4]

        edX = xlf.get_xop_factory_func("ExpandDims")(
            "ed3", [iX], axis=-3, num_newaxis=2
        )

        assert edX.type[0] == "ExpandDims"
        assert edX.attrs["axis"] == -3
        assert edX.attrs["num_newaxis"] == 2
        assert edX.shapes == [1, 1, -1, 4]

        with self.assertRaises(AssertionError):
            edX = xlf.get_xop_factory_func("ExpandDims")(
                "ed4", [iX], axis=-4, num_newaxis=2
            )

    def test_multiply_layer(self):

        iX1 = XLayer(
            type=["Input"],
            name="in1",
            shapes=[-1, 2, 1, 4],
            sizes=[8],
            bottoms=[],
            tops=[],
            targets=[],
        )

        iX2 = XLayer(
            type=["Input"],
            name="in2",
            shapes=[-1, 2, 1, 4],
            sizes=[8],
            bottoms=[],
            tops=[],
            targets=[],
        )

        mX = xlf.get_xop_factory_func("Multiply")("mul2", [iX1, iX2])

        assert mX.type[0] == "Multiply"
        assert mX.shapes == [-1, 2, 1, 4]

        iX3 = XLayer(
            type=["Input"],
            name="in3",
            shapes=[-1, 1, 4, 1],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        mX = xlf.get_xop_factory_func("Multiply")("mul3", [iX1, iX3])

        assert mX.type[0] == "Multiply"
        assert mX.shapes == [-1, 2, 4, 4]

        iX4 = XLayer(
            type=["Input"],
            name="in4",
            shapes=[4, 1],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        mX = xlf.get_xop_factory_func("Multiply")("mul4", [iX1, iX4])

        assert mX.type[0] == "Multiply"
        assert mX.shapes == [-1, 2, 4, 4]

    def test_relu_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        rX = px.ops.relu("relu1", [iX])

        assert rX.type[0] == "ReLU"
        assert rX.shapes == [1, 2, 4, 4]

        from pyxir.graph.ops.l1_basic_nn import relu_transpose_transform

        relu_transpose_transform(rX, axes=[0, 2, 3, 1])

        assert rX.type[0] == "ReLU"
        assert rX.shapes == [1, 4, 4, 2]

    def test_scaling_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        gX = XLayer(
            type=["Constant"],
            name="gamma",
            shapes=[2],
            sizes=[2],
            data=[np.array([1.0, 2.0])],
            bottoms=[],
            tops=[],
            targets=[],
        )

        bX = XLayer(
            type=["Constant"],
            name="beta",
            shapes=[2],
            sizes=[2],
            data=[np.array([1.0, -2.0])],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = xlf.get_xop_factory_func("Scale")("scale1", iX, gX, bX, axis=1)

        assert sX.type[0] == "Scale"
        assert sX.attrs["axis"] == 1

        np.testing.assert_array_equal(sX.data.gamma, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(sX.data.beta, np.array([1.0, -2.0]))

        from pyxir.graph.ops.l1_basic_nn import scale_transpose_transform

        scale_transpose_transform(sX, axes=[0, 2, 3, 1])

        assert sX.type[0] == "Scale"
        assert sX.shapes == [1, 4, 4, 2]
        assert sX.attrs["axis"] == 3

    def test_sigmoid_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        X = xlf.get_xop_factory_func("Sigmoid")("sig1", [iX])

        assert X.type[0] == "Sigmoid"
        assert X.shapes == [1, 2, 4, 4]
        assert X.sizes == [32]

    def test_softmax_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        X = xlf.get_xop_factory_func("Softmax")("soft1", [iX])

        assert X.type[0] == "Softmax"
        assert X.shapes == [1, 2, 4, 4]
        assert X.sizes == [32]

    def test_sqrt_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        X = xlf.get_xop_factory_func("Sqrt")("sqrt1", [iX])

        assert X.type[0] == "Sqrt"
        assert X.shapes == [1, 2, 4, 4]
        assert X.sizes == [32]

    def test_tanh_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        X = xlf.get_xop_factory_func("Tanh")("tanh1", [iX])

        assert X.type[0] == "Tanh"
        assert X.shapes == [1, 2, 4, 4]
        assert X.sizes == [32]
