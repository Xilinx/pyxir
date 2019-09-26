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
Module for testing the relay pyxir frontend


"""

import unittest
import numpy as np

try:
    # ! To import tvm
    import pyxir.frontend.tvm

    import tvm
    from tvm import relay
    from tvm.relay import testing

    from pyxir.frontend.tvm import relay as xf_relay

    skip = False
except Exception as e:
    skip = True


class TestRelayL1BasicConversions(unittest.TestCase):

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_add_biasadd(self):
        left = relay.var(
            "left",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        right = relay.expr.const(np.array([1.0, -1.0], dtype=np.float32))

        net = relay.add(left, right)

        net = relay.Function([left], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        assert layers[1].type[0] == 'BiasAdd'
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert layers[1].bottoms == ['left']
        assert 'relay_id' in layers[1].attrs
        assert layers[1].attrs['axis'] == 3

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_add(self):
        left = relay.var(
            "left",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        right = relay.expr.const(np.zeros((2, 2), dtype=np.float32))

        net = relay.add(left, right)

        net = relay.Function([left], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        assert layers[1].type[0] == 'Constant'
        assert layers[1].tops[0][:3] == 'add'
        assert 'relay_id' in layers[1].attrs

        assert layers[2].type[0] == 'Add'
        assert layers[2].shapes == [-1, 4, 2, 2]
        assert 'relay_id' in layers[2].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_exp(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.exp(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Exp'
        assert 'relay_id' in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_expand_dims(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4), "float32")
        )

        net = relay.expand_dims(data, axis=1, num_newaxis=2)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'ExpandDims'
        assert 'relay_id' in layers[1].attrs
        assert layers[1].attrs['axis'] == 1
        assert layers[1].attrs['num_newaxis'] == 2
        assert layers[1].shapes == [-1, 1, 1, 4]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_log(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.log(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Log'
        assert 'relay_id' in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_rsqrt(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.rsqrt(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'rSqrt'
        assert 'relay_id' in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_sigmoid(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.sigmoid(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Sigmoid'
        assert 'relay_id' in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_softmax(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.nn.softmax(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Softmax'
        assert 'relay_id' in layers[1].attrs
        assert 'axis' in layers[1].attrs
        assert layers[1].attrs['axis'] == -1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_sqrt(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.sqrt(data)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Sqrt'
        assert 'relay_id' in layers[1].attrs
