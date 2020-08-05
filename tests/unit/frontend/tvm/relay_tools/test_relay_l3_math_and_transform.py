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

from pyxir.shapes import TensorShape, TupleShape


class TestRelayL3MathAndTransform(unittest.TestCase):

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_split_int(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 6, 4, 4), "float32")
        )

        net = relay.split(data, indices_or_sections=3, axis=1).astuple()

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Split'
        assert 'relay_id' in layers[1].attrs
        assert layers[1].attrs['axis'] == 1
        assert layers[1].attrs['indices'] == 3
        assert layers[1].shapes == TupleShape([TensorShape([-1, 2, 4, 4]),
                                               TensorShape([-1, 2, 4, 4]),
                                               TensorShape([-1, 2, 4, 4])])

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_split_tuple(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 5, 4, 4), "float32")
        )

        net = relay.split(data, indices_or_sections=(1, 4), axis=1)\
                   .astuple()

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Split'
        assert 'relay_id' in layers[1].attrs
        assert layers[1].attrs['axis'] == 1
        assert layers[1].attrs['indices'] == (1, 4)
        assert layers[1].shapes == TupleShape([TensorShape([-1, 1, 4, 4]),
                                               TensorShape([-1, 3, 4, 4]),
                                               TensorShape([-1, 1, 4, 4])])

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_take(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 3, 224, 224), "float32")
        )

        indices = relay.var(
            "indices",
            relay.TensorType([], "int32")
        )

        net = relay.take(data, indices, axis=1)

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {'indices': np.array(0, np.int32)})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Constant'
        assert layers[1].data == np.array(0, np.int32)
        assert layers[2].type[0] == 'Take'
        assert 'relay_id' in layers[2].attrs
        assert layers[2].attrs['axis'] == 1
        assert layers[2].attrs['mode'] == 'clip'
        assert layers[2].shapes == [-1, 224, 224]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_transpose_constant(self):
        d = np.zeros((1, 3, 2, 2))
        data = relay.var(
            "data",
            relay.TensorType((1, 3, 2, 2), "float32")
        )

        net = relay.transpose(data, axes=(0, 2, 3, 1))

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {'data': d})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Constant'
        assert layers[0].shapes == [1, 2, 2, 3]
        np.testing.assert_array_equal(layers[0].data[0], np.transpose(d, (0, 2, 3, 1)))

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_transpose(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 3, 2, 2), "float32")
        )

        net = relay.transpose(data, axes=(0, 2, 3, 1))

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[0].shapes == [-1, 3, 2, 2]
        assert layers[1].type[0] == 'Transpose'
        assert layers[1].shapes == [-1, 2, 2, 3]
