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
    # Skip TVM tests
    skip = True

if not skip:
    from pyxir.frontend.tvm import relay as xf_relay


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

        func = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

        mod, params = testing.create_workload(simple_net)

        xgraph = xf_relay.from_relay(mod, params)

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

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_simple_network_cvx(self):
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
        simple_net = relay.nn.relu(simple_net)

        simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

        mod, params = testing.create_workload(simple_net)

        xgraph = xf_relay.from_relay(
            mod, params, cvx_preprocessing={"data": "scale-0.5__transpose-2,0,1"}
        )
        
        assert len(xgraph.get_input_names()) == 1
        layers = xgraph.get_layers()

        # assert layers[0].type[0] == "Constant"
        assert layers[0].type[0] == "StrInput"
        assert layers[0].shapes == [-1]
        assert layers[1].type[0] == "Cvx"
        assert layers[1].shapes == [-1, 3, 224, 224]
        assert layers[2].type[0] == "Pad"
        assert layers[3].type[0] == "Convolution"
        assert layers[4].type[0] == "ReLU"

        assert layers[0].tops == ["data_cvx"]
        assert layers[1].bottoms == ["data"]
        assert layers[1].tops[0][:7] == "nn.pad-"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_conv2d_transpose(self):
        data = relay.var("data", relay.TensorType((-1, 1, 3, 3), "float32"))
        weight = relay.var("weight")

        simple_net = relay.nn.conv2d_transpose(
            data=data,
            weight=weight,
            kernel_size=(2, 2),
            channels=1,
            padding=(0, 0),
            strides=(2, 2),
            data_layout="NCHW",
            kernel_layout="IOHW",
        )

        simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)

        mod, params = testing.create_workload(simple_net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[0].shapes == [-1, 1, 3, 3]

        assert layers[1].type[0] == "Conv2DTranspose"
        assert layers[1].shapes == [-1, 1, 6, 6]
        assert layers[1].sizes == [36]
        assert layers[1].attrs["padding"] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert layers[1].attrs["strides"] == [2, 2]
        assert layers[1].attrs["dilation"] == [1, 1]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_resnet_block(self):
        data = relay.var("data", relay.TensorType((-1, 3, 224, 224), "float32"))
        weight = relay.var("weight")
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")

        conv2d0_expr = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        bn0_expr = relay.nn.batch_norm(
            conv2d0_expr, bn_gamma, bn_beta, bn_mmean, bn_mvar
        )[0]
        relu0_expr = relay.nn.relu(bn0_expr)

        max_pool0_expr = relay.nn.max_pool2d(
            relu0_expr, pool_size=(2, 2), strides=(2, 2)
        )

        conv2d1_weight = relay.var("conv2d1_weight")
        conv2d1_bias = relay.var("conv2d1_bias")
        conv2d1_expr = relay.nn.conv2d(
            data=max_pool0_expr,
            weight=conv2d1_weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1),
        )
        bias_add0_expr = relay.nn.bias_add(conv2d1_expr, conv2d1_bias, axis=1)
        relu1_expr = relay.nn.relu(bias_add0_expr)
        add0_expr = relay.op.tensor.add(max_pool0_expr, relu1_expr)

        avg_pool0_expr = relay.nn.avg_pool2d(
            add0_expr, pool_size=(2, 2), strides=(2, 2)
        )
        global_avg_pool0_expr = relay.op.nn.global_avg_pool2d(avg_pool0_expr)
        bf_expr = relay.nn.batch_flatten(global_avg_pool0_expr)

        net = avg_pool0_expr

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Convolution"
        assert layers[2].type[0] == "BatchNorm"
        assert layers[3].type[0] == "ReLU"
        assert layers[4].type[0] == "Pooling"
        assert layers[5].type[0] == "Convolution"
        assert layers[6].type[0] == "BiasAdd"
        assert layers[7].type[0] == "ReLU"
        assert layers[8].type[0] == "Eltwise"
        assert layers[9].type[0] == "Pooling"
        assert layers[9].shapes == [-1, 16, 56, 56]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_inception_block(self):
        data = relay.var("data", relay.TensorType((-1, 3, 224, 224), "float32"))
        weight = relay.var("weight")
        bn_gamma = relay.var("bn_gamma")
        bn_beta = relay.var("bn_beta")
        bn_mmean = relay.var("bn_mean")
        bn_mvar = relay.var("bn_var")

        conv2d0_expr = relay.nn.conv2d(
            data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
        )
        bn0_expr = relay.nn.batch_norm(
            conv2d0_expr, bn_gamma, bn_beta, bn_mmean, bn_mvar
        )[0]
        relu0_expr = relay.nn.relu(bn0_expr)

        max_pool0_expr = relay.nn.max_pool2d(
            relu0_expr, pool_size=(2, 2), strides=(2, 2)
        )

        conv2d1_weight = relay.var("conv2d1_weight")
        conv2d1_bias = relay.var("conv2d1_bias")
        conv2d1_expr = relay.nn.conv2d(
            data=max_pool0_expr,
            weight=conv2d1_weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1),
            strides=(2, 2),
        )
        bias_add1_expr = relay.nn.bias_add(conv2d1_expr, conv2d1_bias, axis=1)
        relu1_expr = relay.nn.relu(bias_add1_expr)

        conv2d2_weight = relay.var("conv2d2_weight")
        conv2d2_bias = relay.var("conv2d2_bias")
        conv2d2_expr = relay.nn.conv2d(
            data=max_pool0_expr,
            weight=conv2d2_weight,
            kernel_size=(3, 3),
            channels=16,
            padding=(1, 1),
            strides=(2, 2),
        )
        bias_add2_expr = relay.nn.bias_add(conv2d2_expr, conv2d2_bias, axis=1)
        relu2_expr = relay.nn.relu(bias_add2_expr)

        concat0_expr = relay.op.tensor.concatenate([relu1_expr, relu2_expr], axis=1)

        global_max_pool0_expr = relay.op.nn.global_max_pool2d(concat0_expr)

        net = global_max_pool0_expr

        net = relay.Function(relay.analysis.free_vars(net), net)

        mod, params = testing.create_workload(net)

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Convolution"
        assert layers[2].type[0] == "BatchNorm"
        assert layers[3].type[0] == "ReLU"
        assert layers[4].type[0] == "Pooling"
        assert layers[5].type[0] == "Convolution"
        assert layers[6].type[0] == "BiasAdd"
        assert layers[7].type[0] == "ReLU"
        assert layers[8].type[0] == "Convolution"
        assert layers[9].type[0] == "BiasAdd"
        assert layers[10].type[0] == "ReLU"
        assert layers[11].type[0] == "Concat"
        assert layers[12].type[0] == "Pooling"
        assert layers[12].shapes == [-1, 32, 1, 1]
