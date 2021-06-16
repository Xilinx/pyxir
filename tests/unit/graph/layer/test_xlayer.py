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
Module for testing the XLayer data structure


"""

import unittest
import numpy as np
import libpyxir as lpx

from pyxir.shapes import TensorShape, TupleShape
from pyxir.shared.vector import IntVector, IntVector2D, FloatVector, StrVector
from pyxir.graph.layer.xlayer import (
    XLayer,
    ConvData,
    ScaleData,
    BatchData,
    defaultXLayer,
)
from pyxir.graph.layer.xattr_dict import XAttrDict


class TestXLayer(unittest.TestCase):
    def test_xlayer_constructor(self):

        X = XLayer()

        assert X.name == ""
        assert X.type == []
        assert X.type == lpx.StrVector([])
        assert X.shapes == TensorShape(lpx.IntVector([]))
        assert X.shapes == []
        assert X.sizes == lpx.IntVector([])
        assert X.sizes == []
        assert X.tops == lpx.StrVector([])
        assert X.tops == []
        assert X.bottoms == lpx.StrVector([])
        assert X.bottoms == []
        assert X.layer == lpx.StrVector([])
        assert X.layer == []
        assert X.data == []
        assert X.targets == lpx.StrVector([])
        assert X.targets == []
        assert X.target == "cpu"
        assert X.subgraph is None
        assert X.internal is False
        assert X.attrs == XAttrDict(lpx.XAttrMap())

    def test_copy(self):
        X = XLayer(
            name="Elon",
            type=["Machine"],
            data=[np.array([1, 2, 3], dtype=np.float32)],
            attrs={"a": [1, 2], "b": -1},
        )

        X_copy = X.copy()
        assert X_copy.name == "Elon"
        assert X_copy.type == ["Machine"]
        np.testing.assert_array_equal(
            X_copy.data[0], np.array([1, 2, 3], dtype=np.float32)
        )
        assert X_copy.attrs["a"] == [1, 2]
        assert X_copy.attrs["b"] == -1

        X.name = "Elon 2"
        X.type[0] = "Robot"
        X.data[0] *= 2
        X.attrs["a"].append(3)

        assert X.name == "Elon-2"
        assert X.type == ["Robot"]
        np.testing.assert_array_equal(X.data[0], np.array([2, 4, 6], dtype=np.float32))
        assert X.attrs["a"] == [1, 2, 3]
        assert X.attrs["b"] == -1

        assert X_copy.name == "Elon"
        assert X_copy.type == ["Machine"]
        np.testing.assert_array_equal(
            X_copy.data[0], np.array([1, 2, 3], dtype=np.float32)
        )
        assert X_copy.attrs["a"] == [1, 2]
        assert X_copy.attrs["b"] == -1

    def test_xlayer_name(self):

        X = XLayer(name="Elon")
        assert X.name == "Elon"

        X2 = X.copy()
        assert X2.name == "Elon"

        X2.name = "Musk"
        assert X.name == "Elon"
        assert X2.name == "Musk"

    def test_xlayer_type(self):

        X = XLayer()

        assert X.type == []
        X.type.append("Iron Man")
        assert X.type == ["Iron Man"]
        assert "Iron Man" in X.type
        X.type[0] = "Iron Man 2"
        assert X.type == ["Iron Man 2"]

        X2 = X.copy()
        assert X2.type == ["Iron Man 2"]
        X2.type[0] = "Iron Man 3"
        assert X.type == ["Iron Man 2"]
        assert X2.type == ["Iron Man 3"]

        X2.type[:] = ["Iron Man 4"]
        assert len(X2.type) == 1

    def test_xlayer_shapes(self):

        # TensorShape
        X = XLayer(shapes=[-1, 2, 4, 4])

        assert X.shapes == [-1, 2, 4, 4]
        assert X.shapes == TensorShape([-1, 2, 4, 4])

        X.shapes[1] = 3
        assert X.shapes == [-1, 3, 4, 4]
        assert X.shapes == TensorShape([-1, 3, 4, 4])

        X.shapes = [-1, 3, 5, 5]
        assert X.shapes == [-1, 3, 5, 5]
        assert X.shapes == TensorShape([-1, 3, 5, 5])
        assert X.shapes.get_size() == [75]

        shapes2 = X.shapes._replace(5, 6)
        assert shapes2 == TensorShape([-1, 3, 6, 6])

        # TupleShape
        X = XLayer(shapes=[[-1, 2, 4, 4], [-1, 2, 3, 3]])

        assert X.shapes == [[-1, 2, 4, 4], [-1, 2, 3, 3]]
        assert X.shapes == TupleShape(
            [TensorShape([-1, 2, 4, 4]), TensorShape([-1, 2, 3, 3])]
        )
        assert X.shapes.get_size() == [32, 18]

        assert X.shapes[0] == [-1, 2, 4, 4]
        assert X.shapes[1] == [-1, 2, 3, 3]
        assert X.shapes[0] == TensorShape([-1, 2, 4, 4])
        assert X.shapes[1] == TensorShape([-1, 2, 3, 3])

        X.shapes[0] = [-1, 1, 2, 2]
        assert X.shapes == [[-1, 1, 2, 2], [-1, 2, 3, 3]]

        X.shapes[0][1] = 3
        assert X.shapes == [[-1, 3, 2, 2], [-1, 2, 3, 3]]
        assert X.shapes.get_size() == [12, 18]
        assert X.shapes.tolist() == [[-1, 3, 2, 2], [-1, 2, 3, 3]]

        X.shapes[1] = [-1, 3, 4, 4]
        assert X.shapes.get_size() == [12, 48]

        shapes2 = X.shapes._replace(4, 6)
        assert shapes2 == [[-1, 3, 2, 2], [-1, 3, 6, 6]]
        assert shapes2 == TupleShape([[-1, 3, 2, 2], [-1, 3, 6, 6]])
        assert shapes2.get_size() == [12, 108]
        assert X.shapes.get_size() == [12, 48]

        # Tuple one element
        X.shapes = [[1, 2, 3, 3]]
        assert X.shapes == [[1, 2, 3, 3]]
        assert X.shapes == TupleShape([[1, 2, 3, 3]])

    def test_xlayer_sizes(self):

        X = XLayer(sizes=[16])

        assert isinstance(X.sizes, IntVector)

        assert X.sizes == [16]
        del X.sizes[0]
        assert X.sizes == []
        X.sizes.append(8)
        assert X.sizes == [8]

        X.sizes = [32]
        assert X.sizes == [32]
        assert len(X.sizes) == 1

    def test_xlayer_tops(self):

        X = XLayer(tops=["Everest"])

        assert isinstance(X.tops, StrVector)

        assert X.tops == ["Everest"]
        assert len(X.tops) == 1
        assert "Everest" in X.tops
        del X.tops[0]
        assert X.tops == []
        X.tops.append("K2")
        assert X.tops == ["K2"]

    def test_xlayer_bottoms(self):

        X = XLayer(bottoms=["Challenger-Deep"])

        assert isinstance(X.bottoms, StrVector)

        assert X.bottoms == ["Challenger-Deep"]
        assert len(X.bottoms) == 1
        assert "Challenger-Deep" in X.bottoms
        del X.bottoms[0]
        assert X.bottoms == []
        assert len(X.bottoms) == 0
        X.bottoms.append("Krubera-Cave")
        assert X.bottoms == ["Krubera-Cave"]

    def test_xlayer_layer(self):

        X = XLayer(layer=["l1"])

        assert isinstance(X.layer, StrVector)

        assert X.layer == ["l1"]
        assert len(X.layer) == 1
        assert "l1" in X.layer
        del X.layer[0]
        assert X.layer == []
        assert len(X.layer) == 0
        X.layer.append("l2")
        assert X.layer == ["l2"]

    def test_xlayer_data(self):

        X = XLayer(data=[np.array([1, 2, 3])])

        assert isinstance(X.data, list)

        assert len(X.data) == 1
        np.testing.assert_array_equal(X.data[0], np.array([1, 2, 3]))

        X.data[0] *= 2
        np.testing.assert_array_equal(X.data[0], np.array([2, 4, 6]))

        X.data = [np.array([3.0, 5.0, 7.0], dtype=np.float32)]
        np.testing.assert_array_equal(
            X.data[0], np.array([3.0, 5.0, 7.0], dtype=np.float32)
        )

        X.data = [
            np.array([2.0, 4.0, 6.0], dtype=np.float32),
            np.array([0, 0], dtype=np.float32),
        ]
        np.testing.assert_array_equal(
            X.data[0], np.array([2.0, 4.0, 6.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(X.data[1], np.array([0.0, 0.0], dtype=np.float32))

        c_data = ConvData(
            weights=np.ones((4, 2, 3, 3), dtype=np.float32),
            biases=np.array([3, 3], dtype=np.float16),
        )
        # print("c_data", c_data)
        X2 = XLayer(
            type=["Convolution"],
            data=[
                np.ones((4, 2, 3, 3), dtype=np.float32) * 2.0,
                np.array([3.0, 3.0], dtype=np.float16),
            ],
        )

        assert isinstance(X2.data, ConvData)
        np.testing.assert_array_equal(X2.data.weights, c_data.weights * 2)
        np.testing.assert_array_equal(X2.data.biases, c_data.biases)

        X2.data = ConvData(
            weights=np.ones((4, 2, 3, 3), dtype=np.float32) * 3,
            biases=np.copy(X2.data.biases),
        )
        np.testing.assert_array_equal(
            X2.data.weights, np.ones((4, 2, 3, 3), dtype=np.float32) * 3
        )
        np.testing.assert_array_equal(
            X2.data.biases, np.array([3, 3], dtype=np.float16)
        )

        # Scale
        X2.type[0] = "Scale"
        X2.data = ScaleData(
            gamma=np.array([1, 2], dtype=np.float32),
            beta=np.array([3, 3], dtype=np.float32),
        )
        assert X2.type == ["Scale"]
        assert isinstance(X2.data, ScaleData)
        np.testing.assert_array_equal(X2.data.gamma, np.array([1, 2], dtype=np.float32))
        np.testing.assert_array_equal(X2.data.beta, np.array([3, 3], dtype=np.float32))

        # BatchData
        X2.type[0] = "BatchNorm"
        X2.data = BatchData(
            mu=np.array([1, 0.5], dtype=np.float32),
            sigma_square=np.array([1, 2], dtype=np.float32),
            gamma=np.array([1, 2], dtype=np.float32),
            beta=np.array([3, 3], dtype=np.float32),
        )
        assert X2.type == ["BatchNorm"]
        assert isinstance(X2.data, BatchData)
        np.testing.assert_array_equal(X2.data.mu, np.array([1, 0.5], dtype=np.float32))
        np.testing.assert_array_equal(
            X2.data.sigma_square, np.array([1, 2], dtype=np.float32)
        )
        np.testing.assert_array_equal(X2.data.gamma, np.array([1, 2], dtype=np.float32))
        np.testing.assert_array_equal(X2.data.beta, np.array([3, 3], dtype=np.float32))

    def test_xlayer_targets(self):

        X = XLayer(targets=["cpu", "dpu"])

        assert isinstance(X.targets, StrVector)

        assert X.targets == ["cpu", "dpu"]
        assert len(X.targets) == 2
        assert "cpu" in X.targets
        assert "dpu" in X.targets
        del X.targets[0]
        with self.assertRaises(IndexError):
            del X.targets[1]
        assert X.targets == ["dpu"]
        assert len(X.targets) == 1
        X.targets.insert(1, "tpu")
        assert X.targets == ["dpu", "tpu"]

    def test_xlayer_target(self):

        X = XLayer(target="dpu")

        assert isinstance(X.target, str)

        assert X.target == "dpu"
        X.target = "cpu"
        assert X.target == "cpu"

    def test_xlayer_subgraph(self):

        X = XLayer(subgraph="xp0")

        assert isinstance(X.target, str)

        assert X.subgraph == "xp0"
        X.subgraph = "xp1"
        assert X.subgraph == "xp1"

    def test_xlayer_subgraph_data(self):

        X_sub = XLayer(name="x_sub", type=["Sub"])

        X = XLayer(subgraph="xp0", subgraph_data=[X_sub])

        assert isinstance(X.target, str)

        sg_data = X.subgraph_data
        assert len(sg_data) == 1
        assert sg_data[0].name == "x_sub"
        assert sg_data[0].type == ["Sub"]

        sg_data[0].type[0] = "Sub2"
        assert sg_data[0].name == "x_sub"
        assert sg_data[0].type == ["Sub2"]

        X.subgraph_data = [sg_data[0], XLayer(name="x_sub3", type=["Sub3"])]
        assert len(X.subgraph_data) == 2
        assert X.subgraph_data[1].name == "x_sub3"
        assert X.subgraph_data[1].type == ["Sub3"]

    def test_xlayer_internal(self):

        X = XLayer(internal=True)

        assert isinstance(X.internal, bool)

        assert X.internal is True
        X.internal = False
        assert not X.internal

    def test_xlayer_attrs(self):
        X = XLayer(attrs={"a": 1})

        assert isinstance(X.attrs, XAttrDict)
        assert X.attrs["a"] == 1
        assert len(X.attrs) == 1

        X.attrs["b"] = [1, 2]
        assert len(X.attrs) == 2
        assert X.attrs["b"] == [1, 2]

        X.attrs["b"] = [-1.5, -2.0]
        assert len(X.attrs) == 2
        assert X.attrs["b"] == [-1.5, -2.0]

        X.attrs.update({"b": ["a", "b"], "c": [[1, 1], [0, 0]]})
        assert len(X.attrs) == 3
        assert X.attrs["b"] == ["a", "b"]
        assert X.attrs["c"] == [[1, 1], [0, 0]]

    def test_to_dict(self):
        X = XLayer(
            name="d",
            type=["Dict"],
            shapes=[1, 3, 224, 224],
            sizes=[3 * 224 * 224],
            tops=["t1", "t2"],
            bottoms=[],
            layer=["d"],
            data=[
                np.array([1, 2, 3], dtype=np.float32),
                np.array([4, 5], dtype=np.float32),
            ],
            targets=["cpu", "dpu"],
            target="cpu",
            subgraph="xp0",
            subgraph_data=[
                XLayer(name="sg", data=[np.array([1.0, 0.0], dtype=np.float32)])
            ],
            internal=False,
            attrs={"a": [1, 2], "b": {"b1": ["b1"], "b2": ["b2", "b2"]}},
        )

        # To dict without data storage (params)
        d = X.to_dict()
        assert d["name"] == "d"
        assert d["type"] == ["Dict"]
        assert d["shapes"] == [1, 3, 224, 224]
        assert d["sizes"] == [3 * 224 * 224]
        assert d["tops"] == ["t1", "t2"]
        assert d["bottoms"] == []
        assert d["layer"] == ["d"]
        assert d["data"] == []
        assert d["targets"] == ["cpu", "dpu"]
        assert d["target"] == "cpu"
        assert d["subgraph"] == "xp0"
        assert len(d["subgraph_data"]) == 1
        assert d["subgraph_data"][0]["name"] == "sg"
        assert d["internal"] is False
        assert len(d["attrs"]) == 2
        assert d["attrs"]["a"] == [1, 2]
        assert d["attrs"]["b"] == {"b1": ["b1"], "b2": ["b2", "b2"]}

        X_c = XLayer.from_dict(d)
        assert X_c.name == "d"
        assert X_c.type == ["Dict"]
        assert X_c.shapes == [1, 3, 224, 224]
        assert X_c.sizes == [3 * 224 * 224]
        assert X_c.tops == ["t1", "t2"]
        assert X_c.bottoms == []
        assert X_c.layer == ["d"]
        assert X_c.data == []
        assert X_c.targets == ["cpu", "dpu"]
        assert X_c.target == "cpu"
        assert X_c.subgraph == "xp0"
        assert len(X_c.subgraph_data) == 1
        assert X_c.subgraph_data[0].name == "sg"
        assert X_c.internal is False
        assert len(X_c.attrs) == 2
        assert X_c.attrs["a"] == [1, 2]
        assert X_c.attrs["b"] == {"b1": ["b1"], "b2": ["b2", "b2"]}

        # To dict with data storage (params)
        d = X.to_dict(data=True)

        np.testing.assert_array_equal(
            d["data"][0], np.array([1, 2, 3], dtype=np.float32)
        )
        np.testing.assert_array_equal(d["data"][1], np.array([4, 5], dtype=np.float32))

        assert d["subgraph_data"][0]["name"] == "sg"
        assert len(d["subgraph_data"][0]["data"]) == 1
        np.testing.assert_array_equal(
            d["subgraph_data"][0]["data"][0], np.array([1.0, 0.0], dtype=np.float32)
        )

        X_c = XLayer.from_dict(d)
        np.testing.assert_array_equal(
            X_c.data[0], np.array([1, 2, 3], dtype=np.float32)
        )
        np.testing.assert_array_equal(X_c.data[1], np.array([4, 5], dtype=np.float32))

        assert X_c.subgraph_data[0].name == "sg"
        assert len(X_c.subgraph_data[0].data) == 1
        np.testing.assert_array_equal(
            X_c.subgraph_data[0].data[0], np.array([1.0, 0.0], dtype=np.float32)
        )
