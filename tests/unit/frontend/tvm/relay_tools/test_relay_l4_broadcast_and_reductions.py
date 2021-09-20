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


class TestRelayL4BroadcastAndReductions(unittest.TestCase):
    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_greater(self):
        left = relay.var("left", relay.TensorType((-1, 4, 2, 2), "float32"))
        right = relay.var("right", relay.TensorType((-1, 4, 2, 2), "float32"))
        g = relay.greater(left, right)
        net = relay.Function([left, right], g)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Input"
        assert layers[2].type[0] == "Greater"
        assert layers[2].shapes == [-1, 4, 2, 2]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_greater_constant(self):
        left = relay.var("left", relay.TensorType((-1, 2, 2, 4), "float32"))
        right = relay.expr.const(np.ones((4,), np.float32))
        g = relay.greater(left, right)
        net = relay.Function([left], g)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Constant"
        assert layers[2].type[0] == "Greater"
        assert layers[2].shapes == [-1, 2, 2, 4]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_mean(self):
        data = relay.var("data", relay.TensorType((-1, 4, 1, 1), "float32"))
        m = relay.mean(data, axis=1)
        net = relay.Function([data], m)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()
        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Mean"
        assert layers[1].shapes == [-1, 1, 1]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_strided_slice(self):
        def _verify(in_shape, begin, end, strides, slice_mode="end", axes=None):
            c = relay.expr.const(np.ones(in_shape, np.float32))
            m = relay.strided_slice(c, begin, end, strides, slice_mode=slice_mode, axes=axes)
            net = relay.Function([], m)
            mod = tvm.IRModule.from_expr(net)
            mod = relay.transform.InferType()(mod)
            expected_shape = [int(e) for e in mod["main"].body.checked_type.shape]

            xgraph = xf_relay.from_relay(mod, {})
            layers = xgraph.get_layers()

            assert layers[0].type[0] == "Constant"
            assert layers[1].type[0] == "StridedSlice"
            assert layers[1].shapes == expected_shape

        _verify((2, 3, 4, 5), (1, 2, 1), (2, 3, 4), (1, 2, 2), axes=(1, 2, 3), slice_mode="size")
        _verify((1, 2, 416, 416), (0, 0, 208, 0), (1, 3, 1000, 1000), (1, 1, 2, 2), slice_mode="size")
        _verify((1, 2, 416, 416), (0, 208, 415), (3, 1000, 1000), (1, 2, 2), axes=(1, 2, 3), slice_mode="size")
        _verify((2, 3, 4, 5), (1, 0, 1), (2, 3, 4), (1, 2, 2), axes=(0, 3, 2), slice_mode="size")
        
        
        _verify((2, 3, 4), (1, 0, 1), (2, 3, 4), (1, 2, 2))
        _verify((2, 3, 4), (1, 2, 1), (2, 3, 4), (1, 2, 2))
        _verify((2, 3, 4), (1, 2, 3), (2, 3, 4), (1, 1, 1))
        _verify((2, 3, 4), (0, 0, 1), (2, 3, 4), (1, 1, 1))
        _verify((2, 3, 4), (0, 0, 1), (2, 3, 4), (2, 3, 4))

        _verify((1, 2, 416, 416), (0, 0, 0, 0), (1, 3, 9223372036854775807, 416), (1, 1, 2, 1))
        _verify((1, 2, 416, 416), (0, 0, 208, 208), (1, 3, 9223372036854775807, 416), (1, 1, 2, 1))
        _verify((1, 2, 416, 416), (0, 0, 208, 0), (1, 3, 9223372036854775807, 1000), (1, 1, 2, 2))

        _verify((2, 3, 4, 5), (1,), (2,), (2,), axes=(2,))
        _verify((2, 3, 4, 5), (0, 1), (2, 3), (1, 2), axes=(0, 2))
        _verify((2, 3, 4, 5), (1, 0, 1), (2, 3, 4), (1, 2, 2), axes=(1, 0, 3))
        _verify((2, 3, 4, 5), (1, 0, 1), (2, 3, 4), (1, 2, 2), axes=(0, 1, 2))

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_strided_slice_onnx(self):
        c = relay.expr.const(np.ones((2, 3, 4), np.float32))
        m = relay.strided_slice(c, (0, 0, 1), (0x7FFFFFFF, 3, 0x7FFFFFFF), (1, 1, 1))
        net = relay.Function([], m)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "StridedSlice"
        assert layers[1].shapes == [2, 3, 3]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_where_constant(self):
        x = relay.var("x", relay.TensorType((-1, 2, 2, 4), "float32"))
        y = relay.var("y", relay.TensorType((-1, 2, 2, 4), "float32"))
        c = relay.expr.const(np.ones((4,), np.float32))
        w = relay.where(c, x, y)
        net = relay.Function([x, y], w)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "Input"
        assert layers[2].type[0] == "Input"
        assert layers[3].type[0] == "AnyOp"
        assert layers[3].shapes == [-1, 2, 2, 4]
