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

"""Module for testing the relay pyxir frontend"""

import unittest
import numpy as np

# ! To import tvm
import pyxir

try:
    import tvm
    from tvm import relay
    from tvm.relay import testing

    skip = False
except Exception as e:
    skip = True

if not skip:
    from pyxir.frontend.tvm import relay as xf_relay

from pyxir.shapes import TupleShape, TensorShape


class TestRelayL1BasicConversions(unittest.TestCase):
    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_add_biasadd(self):
        left = relay.var("left", relay.TensorType((-1, 4, 2, 2), "float32"))

        right = relay.expr.const(np.array([1.0, -1.0], dtype=np.float32))

        net = relay.add(left, right)

        net = relay.Function([left], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "BiasAdd"
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert layers[1].bottoms == ["left"]
        assert "relay_id" in layers[1].attrs
        assert layers[1].attrs["axis"] == 3

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_add(self):
        left = relay.var("left", relay.TensorType((-1, 4, 2, 2), "float32"))
        right = relay.expr.const(np.zeros((2, 2), dtype=np.float32))

        net = relay.add(left, right)
        net = relay.Function([left], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "Constant"
        assert layers[1].tops[0][:3] == "add"
        assert "relay_id" in layers[1].attrs

        assert layers[2].type[0] == "Add"
        assert layers[2].shapes == [-1, 4, 2, 2]
        assert "relay_id" in layers[2].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_batch_norm(self):
        var = relay.var("var", relay.TensorType((-1, 4, 2, 2), "float32"))
        data_mean = relay.expr.const(np.zeros((4,), dtype=np.float32))
        data_var = relay.expr.const(np.ones((4,), dtype=np.float32))
        gamma = relay.expr.const(2.0 * np.ones((4,), dtype=np.float32))
        beta = relay.expr.const(3.0 * np.ones((4,), dtype=np.float32))

        bn = relay.nn.batch_norm(var, gamma, beta, data_mean, data_var)[0]
        # tgi = relay.TupleGetItem(bn, 0)
        func = relay.Function([var], bn)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        bnl = layers[1]
        assert bnl.type[0] == "BatchNorm"
        assert bnl.shapes == [-1, 4, 2, 2]
        np.testing.assert_array_equal(bnl.data[0], np.zeros((4,), dtype=np.float32))
        np.testing.assert_array_equal(bnl.data[1], np.ones((4,), dtype=np.float32))
        np.testing.assert_array_equal(
            bnl.data[2], 2.0 * np.ones((4,), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            bnl.data[3], 3.0 * np.ones((4,), dtype=np.float32)
        )
        assert "relay_id" in bnl.attrs
        assert bnl.attrs["axis"] == 1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_biasadd(self):
        var = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))
        c = relay.expr.const(np.ones((4,), dtype=np.float32))
        ba = relay.nn.bias_add(var, c)

        net = relay.Function([var], ba)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "BiasAdd"
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert layers[1].bottoms == ["data"]
        assert "relay_id" in layers[1].attrs
        assert layers[1].attrs["axis"] == 1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_concatenate(self):
        var1 = relay.var("data1", relay.TensorType((-1, 4, 2, 2), "float32"))
        var2 = relay.var("data2", relay.TensorType((-1, 8, 2, 2), "float32"))
        c = relay.concatenate([var1, var2], axis=1)

        net = relay.Function([var1, var2], c)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 3

        assert layers[0].type[0] == "Input"
        assert layers[0].shapes == [-1, 4, 2, 2]
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "Input"
        assert layers[1].shapes == [-1, 8, 2, 2]
        assert "relay_id" in layers[1].attrs

        assert layers[2].type[0] == "Concat"
        assert layers[2].shapes == [-1, 12, 2, 2]
        assert layers[2].bottoms == ["data1", "data2"]
        assert "relay_id" in layers[2].attrs
        assert layers[2].attrs["axis"] == 1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_dense(self):
        var = relay.var("data", relay.TensorType((-1, 4), "float32"))
        w = relay.expr.const(np.ones((10, 4), dtype=np.float32))
        d = relay.nn.dense(var, w)

        net = relay.Function([var], d)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs
        assert layers[0].shapes == [-1, 4]

        assert layers[1].type[0] == "Dense"
        assert layers[1].shapes == [-1, 10]
        assert layers[1].bottoms == ["data"]
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_nn_dropout(self):
        var = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))
        d = relay.nn.dropout(var)

        net = relay.Function([var], d)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs
        assert layers[0].shapes == [-1, 4, 2, 2]

        assert layers[1].type[0] == "Dropout"
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert layers[1].bottoms == ["data"]
        assert "relay_id" in layers[1].attrs
        assert layers[1].attrs["rate"] == 0.5

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_exp(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.exp(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Exp"
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_expand_dims(self):
        data = relay.var("data", relay.TensorType((-1, 4), "float32"))

        net = relay.expand_dims(data, axis=1, num_newaxis=2)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "ExpandDims"
        assert "relay_id" in layers[1].attrs
        assert layers[1].attrs["axis"] == 1
        assert layers[1].attrs["num_newaxis"] == 2
        assert layers[1].shapes == [-1, 1, 1, 4]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_log(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.log(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Log"
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_multiply(self):
        left = relay.var("left", relay.TensorType((-1, 4, 2, 2), "float32"))

        right = relay.var("right", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.multiply(left, right)

        net = relay.Function([left, right], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "Input"
        assert "relay_id" in layers[1].attrs

        assert layers[2].type[0] == "Multiply"
        assert layers[2].shapes == [-1, 4, 2, 2]
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_multiply_right_constant(self):
        left = relay.var("left", relay.TensorType((-1, 4, 2, 2), "float32"))

        right = relay.expr.const(np.zeros((2, 2), dtype=np.float32))

        net = relay.multiply(left, right)

        net = relay.Function([left], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "Scale"
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_multiply_left_constant(self):
        right = relay.var("right", relay.TensorType((-1, 4, 2, 2), "float32"))
        left = relay.expr.const(np.zeros((2, 2), dtype=np.float32))

        net = relay.multiply(left, right)
        net = relay.Function([right], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        assert layers[1].type[0] == "Scale"
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_relu(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.nn.relu(data)
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "ReLU"
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_rsqrt(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.rsqrt(data)
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "rSqrt"
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_sigmoid(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.sigmoid(data)
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Sigmoid"
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_softmax(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.nn.softmax(data)
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Softmax"
        assert "relay_id" in layers[1].attrs
        assert "axis" in layers[1].attrs
        assert layers[1].attrs["axis"] == -1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_sqrt(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.sqrt(data)
        net = relay.Function(relay.analysis.free_vars(net), net)
        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Sqrt"
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_subtract(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))
        c = relay.expr.const(np.array([1.0, -1.0], dtype=np.float32))

        net = relay.subtract(data, c)
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert len(layers) == 3

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Constant"

        assert layers[2].type[0] == "Sub"
        assert layers[2].shapes == [-1, 4, 2, 2]
        assert "relay_id" in layers[1].attrs
