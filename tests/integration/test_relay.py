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
Module for testing Relay graphs


"""

import unittest
import numpy as np

# ! To import tvm
try:
    import pyxir.frontend.tvm

    import tvm
    from tvm import relay
    from tvm.relay import testing

    from pyxir.frontend.tvm import relay as xf_relay
    skip = False
except Exception as e:
    # Skip TVM tests
    skip = True

from pyxir.target_registry import TargetRegistry

from . import run


class TestRelay(unittest.TestCase):

    target_registry = TargetRegistry()

    # INPUTS/OUTPUTS

    # BASIC NN OPS

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_simple_network(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 1, 4, 4), "float32")
        )
        weight = relay.var("weight")

        # simple_net = relay.nn.pad(data, ((0, 0), (0, 0), (1, 1), (1, 1)))
        simple_net = relay.nn.conv2d(
            data=data,
            weight=weight,
            kernel_size=(2, 2),
            channels=2,
            padding=(0, 0)
        )

        simple_net = relay.Function(
            relay.analysis.free_vars(simple_net),
            simple_net
        )

        mod, params = testing.create_workload(simple_net)

        weight = np.reshape(np.array([[[1, 2], [3, 0]], [[1, 1], [0, 1]]],
                                     dtype=np.float32),
                            (2, 1, 2, 2))

        xgraph = xf_relay.from_relay(mod, {'weight': weight})

        layers = xgraph.get_layers()

        inputs = {
            'data': np.reshape(np.array([
                [10, 10, 0, 40],
                [50, 10, 0, 80],
                [30, 50, 10, 0],
                [10, 90, 30, 40]]), (1, 1, 4, 4))
        }
        res = run._run_network_cpu(xgraph, inputs)
        # print(res[0])

        expected_outpt = np.array([[
            [[180., 40., 80.],
             [160., 160., 190.],
             [160., 340., 100.]],

            [[30., 10., 120.],
             [110., 20., 80.],
             [170., 90., 50.]]
        ]])

        np.testing.assert_array_equal(res[0], expected_outpt)


if __name__ == '__main__':
    unittest.main()
