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


class TestRelayL0Other(unittest.TestCase):

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_var_constant(self):
        var = relay.var(
            "var",
            relay.TensorType((-1, 4, 2, 2), "int64")
        )

        const = relay.expr.const(np.array([1, -1], dtype=np.int64), 'int64')

        net = relay.add(var, const)

        net = relay.Function([var], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert isinstance(layers[0].attrs['dtype'], str)
        assert layers[0].attrs['dtype'] == 'int64'
        assert 'relay_id' in layers[0].attrs

        assert layers[1].type[0] == 'BiasAdd'
        assert layers[1].shapes == [-1, 4, 2, 2]
        assert 'relay_id' in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_relay_op(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.std(data, axis=1, keepdims=False, exclude=False)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'

        assert layers[1].type[0] == 'Mean'
        assert layers[1].shapes == [-1, 1, 2, 2]
        # assert isinstance(layers[1].attrs['relay_id'], list)
        assert layers[1].attrs['axes'] == [1]
        assert layers[1].attrs['keepdims'] is True

        assert layers[2].type[0] == 'RelayOp'
        assert layers[2].shapes == [-1, 2, 2]
        # assert isinstance(layers[2].attrs['relay_id'], list)
        assert layers[2].attrs['relay_shape'] == [-1, 2, 2]
        assert layers[2].attrs['dtype'] == 'float32'
        assert layers[2].attrs['axis'] == '[1]'
        assert layers[2].attrs['keepdims'] == '0'
        assert layers[2].attrs['exclude'] == '0'

        assert layers[3].type[0] == 'Sqrt'
        assert layers[3].shapes == [-1, 2, 2]
        # assert isinstance(layers[3].attrs['relay_id'], list)
