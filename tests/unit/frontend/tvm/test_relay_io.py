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

"""Module for testing the IO for the relay pyxir frontend"""

import os
import json
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
    from pyxir.frontend.tvm.io import load_model_from_file

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestRelayFrontend(unittest.TestCase):
    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_simple_network(self):
        data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
        weight = relay.var("weight")
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")

        simple_net = relay.nn.pad(data, ((0, 0), (0, 0), (1, 1), (1, 1)))
        simple_net = relay.nn.conv2d(
            data=simple_net,
            weight=weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(0, 0),
        )
        simple_net = relay.nn.batch_norm(
            simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar
        )[0]
        simple_net = relay.nn.relu(simple_net)
        simple_net = relay.op.reduce.mean(simple_net, axis=(2, 3))
        simple_net = relay.op.transform.reshape(simple_net, newshape=(1, 16))

        dense_weight = relay.var("dense_weight")
        dense_bias = relay.var("dense_bias")
        simple_net = relay.nn.dense(simple_net, weight=dense_weight, units=10)
        simple_net = relay.nn.bias_add(simple_net, dense_bias, axis=1)

        simple_net = relay.nn.softmax(simple_net, axis=1)
        simple_net = relay.op.transform.reshape(simple_net, newshape=(1, 10))

        simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

        mod, params = testing.create_workload(simple_net)

        json_file = os.path.join(FILE_DIR, "relay_mod_test.json")
        with open(json_file, "w") as fo:
            fo.write(tvm.ir.save_json(mod))

        params_file = os.path.join(FILE_DIR, "relay_params_test.params")
        with open(params_file, "wb") as fo:
            fo.write(relay.save_param_dict(params))

        mod_read, params_read = load_model_from_file("Relay", "Relay")(
            model_path=json_file,
            shapes={"data": [-1, 3, 224, 224]},
            opt_model_path=params_file,
        )

        xgraph = xf_relay.from_relay(mod_read, params_read)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Pad"
        assert layers[2].type[0] == "Convolution"
        assert layers[3].type[0] == "BatchNorm"
        assert layers[4].type[0] == "ReLU"
        assert layers[5].type[0] == "Mean"
        assert layers[6].type[0] == "Reshape"
        assert layers[7].type[0] == "Dense"
        assert layers[8].type[0] == "BiasAdd"
        assert layers[9].type[0] == "Softmax"
        assert layers[10].type[0] == "Reshape"

        os.remove(json_file)
        os.remove(params_file)
