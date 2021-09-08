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
    from tvm.relay.build_module import bind_params_by_name

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
        c = relay.expr.const(np.ones((2, 3, 4), np.float32))
        m = relay.strided_slice(c, (0, 0, 1), (2, 3, 4), (1, 1, 1))
        net = relay.Function([], m)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Constant"
        assert layers[1].type[0] == "StridedSlice"
        assert layers[1].shapes == [2, 3, 3]

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
    def test_strided_slice_dynamic(self):
        def _create_strided_slice(in_shape, begin, end, strides, slice_mode="end", dtype="int32"):
            x = relay.var("x", relay.TensorType(in_shape, "float32"))
            b = relay.var("b", relay.TensorType((len(begin),), dtype))
            end = relay.const(end, dtype=dtype)
            z = relay.strided_slice(x, begin=b, end=end, strides=strides, slice_mode=slice_mode)
            func = relay.Function([x, b], z)
            mod = tvm.IRModule.from_expr(func)
            params = {"b": begin}
            mod["main"] = bind_params_by_name(mod["main"], params)
            return mod
        
        def _test_dyn(in_shape, begin, end, strides, out_shape, slice_mode="end", dtype="int32"):
            mod = _create_strided_slice(in_shape, begin, end, strides, slice_mode, dtype)
            mod = relay.transform.InferType()(mod)
            
            xgraph = xf_relay.from_relay(mod, {})
            layers = xgraph.get_layers()

            assert layers[-1].shapes == out_shape

        def _test_dyn_to_static(in_shape, begin, end, strides, out_shape, slice_mode="end", dtype="int32"):
            mod = _create_strided_slice(in_shape, begin, end, strides, slice_mode, dtype)
            mod = relay.transform.DynamicToStatic()(mod)
            mod = relay.transform.InferType()(mod)

            xgraph = xf_relay.from_relay(mod, {})
            layers = xgraph.get_layers()

            import pdb; pdb.set_trace()

            assert layers[1].type[0] == "StridedSlice"
            assert layers[1].shapes == out_shape
        
        _test_dyn((2, 3, 4), [0, 0, 1], [2, 3, 4], [1, 1, 1], [-1, -1, -1])
        # _test_dyn_to_static((2, 3, 4), [0, 0, 1], [2, 3, 4], [1, 1, 1], [2, 3, 3])
        # _test_dyn_to_static((3, 4, 3), [1, 1, 0], [4, 4, 3], [1, 1, 1], [2, 3, 3])
        # _test_dyn_to_static((2, 3, 4), [0, 0, 1], [0x7FFFFFFF, 3, 0x7FFFFFFF], [1, 1, 1], [2, 3, 3])

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
