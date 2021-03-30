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


class TestL3MathAndTransformations(unittest.TestCase):
    def test_clip_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.clip("clip", iX, a_min=0.0, a_max=10.0)

        assert sX.type[0] == "Clip"
        assert sX.shapes == [1, 2, 4, 4]
        assert sX.sizes == [32]
        assert sX.attrs["a_min"] == 0.0
        assert sX.attrs["a_max"] == 10.0
        assert sX.bottoms == ["in1"]

        from pyxir.graph.ops.l3_math_and_transformations import clip_transpose_transform

        clip_transpose_transform(sX, (0, 2, 3, 1))
        assert sX.type[0] == "Clip"
        assert sX.shapes == [1, 4, 4, 2]
        assert sX.sizes == [32]

    def test_relu6_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.clip("clip1", iX, a_min=0.0, a_max=6.0)

        assert sX.type[0] == "ReLU6"
        assert sX.shapes == [1, 2, 4, 4]
        assert sX.sizes == [32]
        assert sX.attrs == {}
        assert sX.bottoms == ["in1"]

        from pyxir.graph.ops.l3_math_and_transformations import (
            relu6_transpose_transform,
        )

        relu6_transpose_transform(sX, (0, 2, 3, 1))
        assert sX.type[0] == "ReLU6"
        assert sX.shapes == [1, 4, 4, 2]
        assert sX.sizes == [32]

    def test_leaky_relu_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.leaky_relu("leaky_relu", [iX], alpha=0.1)

        assert sX.type[0] == "LeakyReLU"
        assert sX.shapes == [1, 2, 4, 4]
        assert sX.sizes == [32]
        assert sX.attrs == {"alpha": 0.1}
        assert sX.bottoms == ["in1"]

        from pyxir.graph.ops.l3_math_and_transformations import (
            leaky_relu_transpose_transform,
        )

        leaky_relu_transpose_transform(sX, (0, 2, 3, 1))
        assert sX.type[0] == "LeakyReLU"
        assert sX.shapes == [1, 4, 4, 2]
        assert sX.sizes == [32]

    def test_nn_prelu_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.prelu("prelu1", iX, alpha=0.2, axis=1)

        assert sX.type[0] == "pReLU"
        assert sX.shapes == [1, 2, 4, 4]
        assert sX.sizes == [32]
        assert sX.attrs["alpha"] == 0.2
        assert sX.bottoms == ["in1"]

        from pyxir.graph.ops.l3_math_and_transformations import (
            prelu_transpose_transform,
        )

        prelu_transpose_transform(sX, (0, 2, 3, 1))
        assert sX.type[0] == "pReLU"
        assert sX.shapes == [1, 4, 4, 2]
        assert sX.sizes == [32]
        assert sX.attrs["alpha"] == 0.2

    def test_reshape_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 4, 1, 1],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.reshape("reshape1", iX, newshape=[1, 4])

        assert sX.type[0] == "Reshape"
        assert sX.shapes == [1, 4]
        assert sX.sizes == [4]
        assert sX.attrs["shape"] == [1, 4]
        assert sX.bottoms == ["in1"]

    def test_split_layer_int(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 6, 4, 4],
            sizes=[96],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.split("split1", [iX], axis=1, indices=3)

        assert sX.type[0] == "Split"
        assert sX.shapes == TupleShape(
            [
                TensorShape([1, 2, 4, 4]),
                TensorShape([1, 2, 4, 4]),
                TensorShape([1, 2, 4, 4]),
            ]
        )
        assert sX.sizes == [32, 32, 32]
        assert sX.attrs["axis"] == 1
        assert sX.attrs["indices"] == 3
        assert sX.bottoms == ["in1"]

    def test_split_layer_tuple(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 5, 4, 4],
            sizes=[80],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.split("split1", [iX], axis=1, indices=[1, 4])

        assert sX.type[0] == "Split"
        assert sX.shapes == TupleShape(
            [
                TensorShape([1, 1, 4, 4]),
                TensorShape([1, 3, 4, 4]),
                TensorShape([1, 1, 4, 4]),
            ]
        )
        assert sX.sizes == [16, 48, 16]
        assert sX.attrs["axis"] == 1
        assert sX.attrs["indices"] == (1, 4)
        assert sX.bottoms == ["in1"]

    def test_squeeze_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 4, 1, 1],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.squeeze("squeeze1", iX, axis=[2, 3])

        assert sX.type[0] == "Squeeze"
        assert sX.shapes == [1, 4]
        assert sX.sizes == [4]
        assert sX.attrs["axis"] == [2, 3]
        assert sX.bottoms == ["in1"]

    def test_take_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 3, 4, 4],
            sizes=[48],
            bottoms=[],
            tops=[],
            targets=[],
        )

        indX1 = XLayer(
            type=["Constant"],
            name="indices",
            shapes=[],
            sizes=[],
            data=[np.array(0, dtype=np.int32)],
            bottoms=[],
            tops=[],
            targets=[],
        )

        tX = xlf.get_xop_factory_func("Take")("take1", [iX, indX1], axis=1, mode="clip")

        assert tX.type[0] == "Take"
        assert tX.attrs["axis"] == 1
        assert tX.attrs["mode"] == "clip"
        assert tX.bottoms == ["in1", "indices"]
        assert tX.shapes == [1, 4, 4]
        assert tX.sizes == [16]

        indX2 = XLayer(
            type=["Constant"],
            name="indices",
            shapes=[2],
            sizes=[2],
            data=[np.array([0, 2], dtype=np.int32)],
            bottoms=[],
            tops=[],
            targets=[],
        )

        tX = px.ops.take("take2", [iX, indX2], axis=1, mode="clip")

        assert tX.type[0] == "Take"
        assert tX.attrs["axis"] == 1
        assert tX.attrs["mode"] == "clip"
        assert tX.bottoms == ["in1", "indices"]
        assert tX.shapes == [1, 2, 4, 4]
        assert tX.sizes == [32]

    def test_transpose_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.transpose("t1", iX, axes=[0, 2, 3, 1])

        assert sX.type[0] == "Transpose"
        assert sX.shapes == [1, 4, 4, 2]
        assert sX.sizes == [32]
        assert sX.attrs["axes"] == [0, 2, 3, 1]
        assert sX.bottoms == ["in1"]
