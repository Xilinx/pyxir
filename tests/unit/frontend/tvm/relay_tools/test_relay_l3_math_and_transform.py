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

from pyxir.shapes import TensorShape, TupleShape


class TestRelayL3MathAndTransform(unittest.TestCase):
    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_arange(self):
        start = relay.expr.const(1.0)
        stop = relay.expr.const(5.0)
        interval = relay.expr.const(1.5)
        a = relay.arange(start, stop, interval)
        net = relay.Function([], a)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert len(layers) == 4

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [1]

        assert layers[1].type[0] == "Constant"
        assert layers[1].shapes == [1]

        assert layers[2].type[0] == "Constant"
        assert layers[2].shapes == [1]

        assert layers[3].type[0] == "AnyOp"
        assert layers[3].shapes == [3]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_cast(self):
        data = relay.var("data", relay.TensorType((-1, 6, 4, 4), "float32"))

        net = relay.cast(data, dtype="int8")
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Cast"
        assert layers[1].attrs["dtype"] == "int8"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_clip(self):
        data = relay.var("data", relay.TensorType((-1, 6, 4, 4), "float32"))

        net = relay.clip(data, 0.0, 7.0)
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Clip"
        assert layers[1].attrs["a_min"] == 0.0
        assert layers[1].attrs["a_max"] == 7.0

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_ones_like(self):
        c = relay.expr.const(np.ones((1, 6, 4, 4), np.float32))
        net = relay.ones_like(c)
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "AnyOp"
        assert layers[1].shapes == [1, 6, 4, 4]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_leaky_relu(self):
        data = relay.var("data", relay.TensorType((-1, 6, 4, 4), "float32"))
        net = relay.nn.leaky_relu(data, alpha=0.1)
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "LeakyReLU"
        assert layers[1].attrs["alpha"] == 0.1

    # @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    # def test_prelu(self):
    #     data = relay.var("data", relay.TensorType((-1, 2, 4, 4), "float32"))
    #     c = relay.expr.const(np.array([0.1, 0.2], dtype=np.float32))
    #     net = relay.nn.prelu(data, alpha=c)
    #     net = relay.Function([data], net)
    #     mod = tvm.IRModule.from_expr(net)
    #     mod = relay.transform.InferType()(mod)

    #     xgraph = xf_relay.from_relay(mod, {})
    #     layers = xgraph.get_layers()

    #     assert layers[0].type[0] == 'Input'
    #     assert layers[1].type[0] == 'pReLU'
    #     assert layers[1].attrs['alpha'] == .1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_repeat(self):
        c = relay.expr.const(np.ones((2, 2), dtype=np.float32))
        net = relay.repeat(c, repeats=2, axis=0)
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "AnyOp"
        assert layers[1].shapes == [8]

        c = relay.expr.const(np.ones((2, 2), dtype=np.float32))
        net = relay.repeat(c, repeats=2, axis=1)
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "AnyOp"
        assert layers[1].shapes == [2, 4]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_reshape(self):
        # 1
        c = relay.expr.const(np.ones((2, 3, 4), dtype=np.float32))
        net = relay.reshape(c, (4, 0, 2))
        net = relay.Tuple([net])
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [4, 3, 2]

        # 2
        c = relay.expr.const(np.ones((2, 3, 4), dtype=np.float32))
        net = relay.reshape(c, (6, 1, -1))
        net = relay.Tuple([net])
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [6, 1, 4]

        # 3
        c = relay.expr.const(np.ones((2, 3, 4), dtype=np.float32))
        net = relay.reshape(c, (-2,))
        net = relay.Tuple([net])
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [2, 3, 4]

        # 4
        c = relay.expr.const(np.ones((2, 3, 4, 5), dtype=np.float32))
        net = relay.reshape(c, (-3, -3))
        net = relay.Tuple([net])
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [6, 20]

        # 5
        data = relay.var("data", relay.TensorType((-1, 6, 1, 1), "float32"))
        net = relay.reshape(data, (-1, 6))
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[0].shapes == [-1, 6, 1, 1]

        assert layers[1].type[0] == "Reshape"
        assert layers[1].shapes == [-1, 6]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_split_int(self):
        data = relay.var("data", relay.TensorType((-1, 6, 4, 4), "float32"))

        net = relay.split(data, indices_or_sections=3, axis=1).astuple()

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Split"
        assert "relay_id" in layers[1].attrs
        assert layers[1].attrs["axis"] == 1
        assert layers[1].attrs["indices"] == 3
        assert layers[1].shapes == TupleShape(
            [
                TensorShape([-1, 2, 4, 4]),
                TensorShape([-1, 2, 4, 4]),
                TensorShape([-1, 2, 4, 4]),
            ]
        )

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_split_tuple(self):
        data = relay.var("data", relay.TensorType((-1, 5, 4, 4), "float32"))

        net = relay.split(data, indices_or_sections=(1, 4), axis=1).astuple()

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Split"
        assert "relay_id" in layers[1].attrs
        assert layers[1].attrs["axis"] == 1
        assert layers[1].attrs["indices"] == (1, 4)
        assert layers[1].shapes == TupleShape(
            [
                TensorShape([-1, 1, 4, 4]),
                TensorShape([-1, 3, 4, 4]),
                TensorShape([-1, 1, 4, 4]),
            ]
        )

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_squeeze(self):
        data = relay.var("data", relay.TensorType((-1, 6, 1, 1), "float32"))
        net = relay.squeeze(data, axis=(2, 3))
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Squeeze"
        assert layers[1].shapes == [-1, 6]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_take(self):
        data = relay.var("data", relay.TensorType((-1, 3, 224, 224), "float32"))

        indices = relay.var("indices", relay.TensorType([], "int32"))

        net = relay.take(data, indices, axis=1)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {"indices": np.array(0, np.int32)})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Constant"
        assert layers[1].data == np.array(0, np.int32)
        assert layers[2].type[0] == "Take"
        assert "relay_id" in layers[2].attrs
        assert layers[2].attrs["axis"] == 1
        assert layers[2].attrs["mode"] == "clip"
        assert layers[2].shapes == [-1, 224, 224]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_transpose_constant(self):
        d = np.zeros((1, 3, 2, 2))
        data = relay.var("data", relay.TensorType((1, 3, 2, 2), "float32"))

        net = relay.transpose(data, axes=(0, 2, 3, 1))
        net = relay.Tuple([net])
        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {"data": d})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [1, 2, 2, 3]
        np.testing.assert_array_equal(layers[0].data[0], np.transpose(d, (0, 2, 3, 1)))

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_transpose(self):
        data = relay.var("data", relay.TensorType((-1, 3, 2, 2), "float32"))

        net = relay.transpose(data, axes=(0, 2, 3, 1))

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[0].shapes == [-1, 3, 2, 2]
        assert layers[1].type[0] == "Transpose"
        assert layers[1].shapes == [-1, 2, 2, 3]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_zeros_like(self):
        c = relay.expr.const(np.ones((1, 6, 4, 4), np.float32))
        net = relay.zeros_like(c)
        net = relay.Function([], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "AnyOp"
        assert layers[1].shapes == [1, 6, 4, 4]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_arange_full_and_reshape(self):
        start = relay.expr.const(0.0)
        stop = relay.expr.const(10.0)
        step = relay.expr.const(1.0)

        fill_val = relay.expr.const(1.0)
        fill_shape = [10, 1]
        dtype = "float32"

        left = relay.arange(start, stop, step, dtype)
        left = relay.reshape(left, [-1, 1])
        left = relay.reshape(left, [1, -1])

        right = relay.full(fill_val, fill_shape, dtype)
        right = relay.reshape(right, [1, -1])

        net = relay.multiply(left, right)

        mod = tvm.IRModule.from_expr(net)
        params = {}
        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert len(layers) == 10
        assert layers[0].type[0] == "Constant"
        assert layers[3].type[0] == "AnyOp"
        assert layers[7].type[0] == "AnyOp"
        assert layers[5].shapes == [1, 10]
        assert layers[8].shapes == [1, 10]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_full(self):
        fill_val = relay.expr.const(1.0)
        fill_shape = [10, 1]

        net = relay.full(fill_val, fill_shape, "float32")
        net = relay.reshape(net, [1, -1])
        mod = tvm.IRModule.from_expr(net)
        params = {}
        xgraph = xf_relay.from_relay(mod, params)
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[0].shapes == [1]
        assert layers[1].type[0] == "AnyOp"
        assert layers[1].shapes == [10, 1]
        assert layers[2].type[0] == "Reshape"
        assert layers[2].shapes == [1, 10]
