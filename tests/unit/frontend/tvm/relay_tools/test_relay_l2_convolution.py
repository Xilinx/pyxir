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


class TestRelayL2Convolutions(unittest.TestCase):

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_nn_upsampling(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.nn.upsampling(data, scale_h=3, scale_w=2)

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        params = {}

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Upsampling2D'
        assert 'relay_id' in layers[1].attrs
        assert layers[1].shapes == [-1, 4, 6, 4]
        assert layers[1].attrs['scale_h'] == 3
        assert layers[1].attrs['scale_w'] == 2
        assert layers[1].attrs['data_layout'] == 'NCHW'
        assert layers[1].attrs['method'] == 'nearest_neighbor'
        assert layers[1].attrs['align_corners'] is False
