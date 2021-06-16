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

"""Module for testing the xgraph functionality"""

import os
import unittest
import numpy as np

# ! Important for device registration
import pyxir as px

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise unittest.SkipTest(
        "Skipping Quantization Tensorflow related test because Tensorflow"
        " is not available"
    )

from pyxir import partition
from pyxir.targets import qsim
from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.xgraph import XGraph

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestXGraph(unittest.TestCase):
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
        target_registry.register_target(
            "test",
            xgraph_optimizer,
            xgraph_quantizer,
            xgraph_compiler,
            xgraph_build_func,
        )

        @register_op_support_check("test", "Convolution")
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check("test", "Pooling")
        def pooling_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):
        target_registry = TargetRegistry()
        target_registry.unregister_target("test")
        target_registry.unregister_target("qsim")

    def test_xgraph_meta_attrs(self):

        xgraph = XGraph("xg")
        assert len(xgraph.meta_attrs) == 0

        xgraph.meta_attrs["test_attr"] = "test_val"
        assert len(xgraph.meta_attrs) == 1
        assert xgraph.meta_attrs["test_attr"] == "test_val"

        xgraph.meta_attrs["test_attr2"] = {"test_key": "test_val"}
        assert len(xgraph.meta_attrs) == 2
        assert xgraph.meta_attrs["test_attr2"] == {"test_key": "test_val"}

        xgraph.meta_attrs = {"d_test_attr": ["t1", "t2"]}
        assert len(xgraph.meta_attrs) == 1
        assert xgraph.meta_attrs["d_test_attr"] == ["t1", "t2"]

    def test_xgraph_add_get(self):
        def _test_add_get(in_name: str, conv_name: str):

            expected_in_name = px.stringify(in_name)
            expected_conv_name = px.stringify(conv_name)

            in1 = px.ops.input(op_name=expected_in_name, shape=[1, 2, 4, 4])
            W = px.ops.constant("W", np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
            X_conv = px.ops.conv2d(
                op_name=conv_name, input_layer=in1, weights_layer=W, kernel_size=[2, 2]
            )

            xgraph = XGraph()
            xgraph.add(in1)

            assert len(xgraph) == 1
            assert len(xgraph.get_layer_names()) == 1
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

            assert isinstance(xgraph.get(in_name), XLayer)
            assert xgraph.get(in_name).bottoms == []
            assert xgraph.get(in_name).tops == []

            xgraph.add(X_conv)

            assert len(xgraph) == 2
            assert xgraph.get_layer_names() == [expected_in_name, expected_conv_name]
            assert xgraph.get_output_names() == [expected_conv_name]
            assert xgraph.get_input_names() == [expected_in_name]

            assert xgraph.get(in_name).tops == [expected_conv_name]

            assert isinstance(xgraph.get(conv_name), XLayer)
            assert xgraph.get(conv_name).bottoms == [expected_in_name]
            assert xgraph.get(conv_name).tops == []
            assert xgraph.get(conv_name).type == ["Convolution"]

            np.testing.assert_array_equal(
                xgraph.get(conv_name).data.weights,
                np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                xgraph.get(conv_name).data.biases, np.array([0.0], dtype=np.float32),
            )

            xgraph.get(conv_name).data = ConvData(
                weights=xgraph.get(conv_name).data.weights * 2,
                biases=xgraph.get(conv_name).data.biases,
            )
            np.testing.assert_array_equal(
                xgraph.get(conv_name).data.weights,
                np.array([[[[2, 4], [6, 8]]]], dtype=np.float32),
            )

            xgraph.remove(X_conv.name)

            assert len(xgraph) == 1
            assert in_name in xgraph
            assert len(xgraph.get_layer_names()) == 1
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

        _test_add_get("in1", "conv1")
        _test_add_get("in:1", "conv::1")

    def test_xgraph_add_remove(self):
        def _test_add_remove(in_name: str, conv_name: str):
            in1 = px.ops.input(op_name=in_name, shape=[1, 2, 4, 4])
            W = px.ops.constant("W", np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
            X_conv = px.ops.conv2d(
                op_name=conv_name, input_layer=in1, weights_layer=W, kernel_size=[2, 2]
            )

            xgraph = XGraph()
            xgraph.add(in1)

            assert len(xgraph) == 1
            assert len(xgraph.get_layer_names()) == 1
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

            xgraph.add(X_conv)

            assert len(xgraph) == 2
            assert len(xgraph.get_layer_names()) == 2
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

            xgraph.remove(X_conv.name)

            assert len(xgraph) == 1
            assert len(xgraph.get_layer_names()) == 1
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

        _test_add_remove("in1", "conv1")
        _test_add_remove("in:1", "conv::1")

    def test_xgraph_insert(self):
        def _test_xgraph_insert(
            in_name: str,
            in2_name: str,
            conv_name: str,
            pool_name: str,
            add_name: str,
            conv2_name: str,
        ):

            expected_in_name = px.stringify(in_name)
            expected_in2_name = px.stringify(in2_name)
            expected_conv_name = px.stringify(conv_name)
            expected_pool_name = px.stringify(pool_name)
            expected_add_name = px.stringify(add_name)
            expected_conv2_name = px.stringify(conv2_name)

            in1 = px.ops.input(op_name=in_name, shape=[1, 2, 4, 4])
            W = px.ops.constant("W", np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
            X_conv = px.ops.conv2d(
                op_name=conv_name, input_layer=in1, weights_layer=W, kernel_size=[2, 2]
            )
            # X_pool = px.ops.pool2d(op_name=pool_name, input_layer=X_conv, )

            xgraph = XGraph()
            xgraph.add(in1)

            assert len(xgraph) == 1
            assert len(xgraph.get_layer_names()) == 1
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

            xgraph.add(X_conv)

            assert len(xgraph) == 2
            assert len(xgraph.get_layer_names()) == 2
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

            X_pool = XLayer(
                name=pool_name,
                type=["Pooling"],
                bottoms=[in_name],
                tops=[conv_name],
                targets=[],
            )
            xgraph.insert(X_pool)

            assert len(xgraph) == 3
            assert len(xgraph.get_layer_names()) == 3
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 1

            xlayers = xgraph.get_layers()
            assert xlayers[0].name == expected_in_name
            assert xlayers[0].bottoms == []
            assert xlayers[0].tops == [expected_pool_name]
            assert xlayers[1].name == expected_pool_name
            assert xlayers[1].bottoms == [expected_in_name]
            assert xlayers[1].tops == [expected_conv_name]
            assert xlayers[2].name == expected_conv_name
            assert xlayers[2].bottoms == [expected_pool_name]
            assert xlayers[2].tops == []

            X_in2 = px.ops.input(op_name=in2_name, shape=[1, 2, 4, 4])
            xgraph.add(X_in2)

            X_add = XLayer(
                name=add_name,
                type=["Eltwise"],
                bottoms=[conv_name, in2_name],
                tops=[],
                targets=[],
            )
            xgraph.add(X_add)

            X_conv2 = XLayer(
                name=conv2_name,
                type=["Convolution"],
                bottoms=[in2_name],
                tops=[add_name],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0.0, 1.0], dtype=np.float32),
                ),
                targets=[],
            )
            xgraph.insert(X_conv2)

            assert len(xgraph) == 6
            assert len(xgraph.get_layer_names()) == 6
            assert len(xgraph.get_output_names()) == 1
            assert len(xgraph.get_input_names()) == 2

            xlayers = xgraph.get_layers()

            assert xlayers[0].name == expected_in_name
            assert xlayers[0].bottoms == []
            assert xlayers[0].tops == [expected_pool_name]
            assert xlayers[1].name == expected_pool_name
            assert xlayers[1].bottoms == [expected_in_name]
            assert xlayers[1].tops == [expected_conv_name]
            assert xlayers[2].name == expected_conv_name
            assert xlayers[2].bottoms == [expected_pool_name]
            assert xlayers[2].tops == [expected_add_name]
            assert xlayers[3].name == expected_in2_name
            assert xlayers[3].bottoms == []
            assert xlayers[3].tops == [expected_conv2_name]
            assert xlayers[4].name == expected_conv2_name
            assert xlayers[4].bottoms == [expected_in2_name]
            assert xlayers[4].tops == [expected_add_name]
            assert xlayers[5].name == expected_add_name
            assert xlayers[5].bottoms == [expected_conv_name, expected_conv2_name]
            assert xlayers[5].tops == []

        _test_xgraph_insert(
            in_name="in1",
            conv_name="conv1",
            pool_name="pool1",
            add_name="add1",
            in2_name="in2",
            conv2_name="conv2",
        )
        _test_xgraph_insert(
            in_name="in:1",
            conv_name="conv::1",
            pool_name="pool:1",
            add_name="add: 1",
            in2_name="in2:",
            conv2_name="conv:2",
        )

    def test_xgraph_device_tagging(self):

        xgraph = XGraph()
        xgraph.add(XLayer(name="in1", type=["Input"], bottoms=[], tops=[], targets=[]))

        xgraph.add(XLayer(name="in2", type=["Input"], bottoms=[], tops=[], targets=[]))

        xgraph.add(
            XLayer(
                name="conv1",
                type=["Convolution"],
                bottoms=["in1"],
                tops=[],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0.0, 1.0], dtype=np.float32),
                ),
                targets=[],
            )
        )

        xgraph.add(
            XLayer(
                name="add1",
                type=["Eltwise"],
                bottoms=["conv1", "in2"],
                tops=[],
                targets=[],
            )
        )

        xgraph.insert(
            XLayer(
                name="conv2",
                type=["Convolution"],
                bottoms=["in2"],
                tops=["add1"],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0.0, 1.0], dtype=np.float32),
                ),
                targets=[],
            )
        )

        xgraph.add(
            XLayer(
                name="pool1", type=["Pooling"], bottoms=["add1"], tops=[], targets=[]
            )
        )
        xgraph = partition(xgraph, ["cpu"])
        assert len(xgraph) == 6
        xlayers = xgraph.get_layers()
        assert xgraph.get_layer_names() == [
            "in1",
            "conv1",
            "in2",
            "conv2",
            "add1",
            "pool1",
        ]
        assert set(xlayers[0].targets) == set(["cpu", "qsim"])
        assert set(xlayers[1].targets) == set(["cpu", "qsim", "test"])
        assert set(xlayers[2].targets) == set(["cpu", "qsim"])
        assert set(xlayers[3].targets) == set(["cpu", "qsim", "test"])
        assert set(xlayers[4].targets) == set(["cpu", "qsim"])
        assert set(xlayers[5].targets) == set(["cpu", "qsim", "test"])

        xgraph.remove("conv1")
        assert len(xgraph) == 5
        xlayers = xgraph.get_layers()

        assert xgraph.get_layer_names() == ["in1", "in2", "conv2", "add1", "pool1"]

        assert xlayers[3].type[0] == "Eltwise"
        assert xlayers[3].bottoms == ["in1", "conv2"]

        assert set(xlayers[0].targets) == set(["cpu", "qsim"])
        assert set(xlayers[1].targets) == set(["cpu", "qsim"])
        assert set(xlayers[2].targets) == set(["cpu", "qsim", "test"])
        assert set(xlayers[3].targets) == set(["cpu", "qsim"])
        assert set(xlayers[4].targets) == set(["cpu", "qsim", "test"])

    def test_copy(self):
        def _test_copy(
            in1_name: str,
            in2_name: str,
            conv1_name: str,
            add_name: str,
            conv2_name: str,
            pool_name: str,
        ):

            expected_in1_name = px.stringify(in1_name)
            expected_in2_name = px.stringify(in2_name)
            expected_conv1_name = px.stringify(conv1_name)
            expected_conv2_name = px.stringify(conv2_name)
            expected_pool_name = px.stringify(pool_name)
            expected_add_name = px.stringify(add_name)

            in1 = px.ops.input(op_name=in1_name, shape=[1, 2, 4, 4])
            in2 = px.ops.input(op_name=in2_name, shape=[1, 2, 4, 4])
            W = px.ops.constant("W", np.array([[[[1, 2], [3, 4]]]], dtype=np.float32))
            X_conv = px.ops.conv2d(
                op_name=conv1_name, input_layer=in1, weights_layer=W, kernel_size=[2, 2]
            )
            X_add = px.ops.eltwise(op_name=add_name, lhs_layer=X_conv, rhs_layer=in2)
            X_conv2 = px.ops.conv2d(
                op_name=conv2_name, input_layer=in2, weights_layer=W, kernel_size=[2, 2]
            )
            X_pool = px.ops.pool2d(
                op_name=pool_name, input_layer=X_add, pool_type="Avg", pool_size=[2, 2]
            )

            xgraph = XGraph()
            xgraph.add(in1)
            xgraph.add(in2)
            xgraph.add(X_conv)
            xgraph.add(X_add)

            xgraph.insert(
                XLayer(
                    name=conv2_name,
                    type=["Convolution"],
                    bottoms=[in2_name],
                    tops=[add_name],
                    data=ConvData(
                        weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                        biases=np.array([0.0, 1.0], dtype=np.float32),
                    ),
                    targets=[],
                )
            )

            xgraph.add(X_pool)

            assert len(xgraph) == 6
            assert xgraph.get_layer_names() == [
                expected_in1_name,
                expected_conv1_name,
                expected_in2_name,
                expected_conv2_name,
                expected_add_name,
                expected_pool_name,
            ]

            xg_copy = xgraph.copy()
            assert len(xg_copy) == 6
            assert xg_copy.get_layer_names() == [
                expected_in1_name,
                expected_conv1_name,
                expected_in2_name,
                expected_conv2_name,
                expected_add_name,
                expected_pool_name,
            ]
            xgc_layers = xg_copy.get_layers()

            assert xgc_layers[1].type == ["Convolution"]
            assert xg_copy.get(conv1_name).type == ["Convolution"]

            xgc_layers[1].type = ["Convolution2"]
            assert xg_copy.get(conv1_name).type == ["Convolution2"]

            xgc_layers[1].type = ["Convolution"]
            assert xgc_layers[1].type == ["Convolution"]
            assert xg_copy.get(conv1_name).type == ["Convolution"]

            np.testing.assert_array_equal(
                xgc_layers[1].data.weights,
                np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                xgc_layers[1].data.biases, np.array([0.0], dtype=np.float32)
            )

            xgraph.get(conv1_name).data = ConvData(
                weights=xgc_layers[1].data.weights * 2, biases=xgc_layers[1].data.biases
            )

            np.testing.assert_array_equal(
                xgraph.get(conv1_name).data.weights,
                np.array([[[[2, 4], [6, 8]]]], dtype=np.float32),
            )

            np.testing.assert_array_equal(
                xgc_layers[1].data.weights,
                np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                xgc_layers[1].data.biases, np.array([0.0], dtype=np.float32)
            )

        _test_copy(
            in1_name="in1",
            in2_name="in2",
            conv1_name="conv1",
            conv2_name="conv2",
            add_name="add1",
            pool_name="pool1",
        )
        _test_copy(
            in1_name="in:1",
            in2_name="in::2",
            conv1_name="conv:1",
            conv2_name="conv::2",
            add_name="add 1",
            pool_name="pool 1",
        )

    def test_visualize(self):

        xgraph = XGraph()
        xgraph.add(XLayer(name="in1", type=["Input"], bottoms=[], tops=[], targets=[]))

        xgraph.add(XLayer(name="in2", type=["Input"], bottoms=[], tops=[], targets=[]))

        xgraph.add(
            XLayer(
                name="conv1",
                type=["Convolution"],
                bottoms=["in1"],
                tops=[],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0.0, 1.0], dtype=np.float32),
                ),
                targets=[],
            )
        )

        xgraph.add(
            XLayer(
                name="add1",
                type=["Eltwise"],
                bottoms=["conv1", "in2"],
                tops=[],
                targets=[],
            )
        )

        xgraph.insert(
            XLayer(
                name="conv2",
                type=["Convolution"],
                bottoms=["in2"],
                tops=["add1"],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0.0, 1.0], dtype=np.float32),
                ),
                targets=[],
            )
        )

        xgraph.add(
            XLayer(
                name="conv3",
                type=["Convolution"],
                bottoms=["add1"],
                tops=[],
                data=ConvData(
                    weights=np.array([[[[1, 2], [3, 4]]]], dtype=np.float32),
                    biases=np.array([0.0, 1.0], dtype=np.float32),
                ),
                targets=[],
            )
        )

        xgraph.add(
            XLayer(
                name="pool1", type=["Pooling"], bottoms=["add1"], tops=[], targets=[]
            )
        )

        xgraph.add(
            XLayer(
                name="add2",
                type=["Eltwise"],
                bottoms=["conv3", "pool1"],
                tops=[],
                targets=[],
            )
        )

        assert len(xgraph) == 8
        assert xgraph.get_layer_names() == [
            "in1",
            "conv1",
            "in2",
            "conv2",
            "add1",
            "conv3",
            "pool1",
            "add2",
        ]

        out_file = os.path.join(FILE_DIR, "viz.png")
        xgraph.visualize(out_file)

        os.remove(out_file)
