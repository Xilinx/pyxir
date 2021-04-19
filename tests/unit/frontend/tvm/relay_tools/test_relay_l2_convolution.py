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
import logging

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

logging.basicConfig()
logger = logging.getLogger("pyxir")
# logger.setLevel(logging.DEBUG)


def conv2d_test_util(
    in_shape,
    weight_shape,
    out_shape,
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    groups=1,
    data_layout="NCHW",
    kernel_layout="OIHW",
    conv_transpose=False,
):
    data = relay.var("data", relay.TensorType(in_shape, "float32"))
    weight = relay.expr.const(np.ones(weight_shape, dtype=np.float32))
    
    if not conv_transpose:
        c = relay.nn.conv2d(
            data=data,
            weight=weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )
        expected_type = "Convolution"
        out_channel = weight_shape[kernel_layout.index("O")]
    else:
        c = relay.nn.conv2d_transpose(
            data=data,
            weight=weight,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )
        expected_type = "Conv2DTranspose"
        out_channel = weight_shape[kernel_layout.index("I")]

    func = relay.Function([data], c)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)

    xg = xf_relay.from_relay(mod, {})
    layers = xg.get_layers()
    assert len(layers) == 2

    assert layers[0].type[0] == "Input"
    assert "relay_id" in layers[0].attrs

    X = layers[1]
    assert X.type[0] == expected_type
    assert X.shapes == list(
        out_shape
    ), "Expected out shape: {0}, but got: {1}".format(out_shape, X.shapes)

    assert "relay_id" in X.attrs
    assert X.attrs["kernel_size"] == [
        weight_shape[data_layout.index("H")],
        weight_shape[data_layout.index("W")],
    ]
    assert X.attrs["strides"] == list(strides)
    expected_padding = [
        [0, 0],
        [0, 0],
        [padding[0], padding[2]],
        [padding[1], padding[3]],
    ]
    assert (
        X.attrs["padding"] == expected_padding
    ), "Expected padding: {0}, but got: {1}".format(
        expected_padding, X.attrs["padding"]
    )
    assert X.attrs["channels"] == [
        in_shape[data_layout.index("C")],
        out_channel,
    ]
    assert X.attrs["data_layout"] == data_layout
    assert X.attrs["kernel_layout"] == "OIHW"
    assert X.attrs["groups"] == groups


class TestRelayL2Convolutions(unittest.TestCase):
    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_avg_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 5, 5), "float32"))
        avg_pool = relay.nn.avg_pool2d(
            var,
            pool_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1),
            ceil_mode=True,
            count_include_pad=True,
        )

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        X = layers[1]
        assert X.type[0] == "Pooling"
        assert X.shapes == [-1, 2, 3, 3]
        assert "relay_id" in X.attrs
        assert X.attrs["padding"] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs["insize"] == [5, 5]
        assert X.attrs["outsize"] == [3, 3]
        assert X.attrs["data_layout"] == "NCHW"
        assert X.attrs["strides"] == [2, 2]
        assert X.attrs["kernel_size"] == [3, 3]
        assert X.attrs["pool_type"] == "Avg"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_batch_flatten(self):
        data = relay.var("data", relay.TensorType((-1, 1, 1, 4), "float32"))

        net = relay.nn.batch_flatten(data)
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"

        assert layers[1].type[0] == "Flatten"
        assert layers[1].shapes == [-1, 4]
        assert "relay_id" in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_conv2d(self):
        conv2d_test_util(
            in_shape=(-1, 1, 4, 4),
            weight_shape=(2, 1, 2, 2),
            out_shape=(-1, 2, 3, 3),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        conv2d_test_util(
            in_shape=(-1, 1, 4, 4),
            weight_shape=(1, 2, 2, 2),
            out_shape=(-1, 2, 3, 3),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="IOHW",
        )
        conv2d_test_util(
            in_shape=(1, 256, 28, 28),
            weight_shape=(256, 256, 3, 3),
            out_shape=(-1, 256, 28, 28),
            padding=(2, 2, 2, 2),
            dilation=(2, 2),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        conv2d_test_util(
            in_shape=(1, 256, 28, 28),
            weight_shape=(256, 256, 3, 3),
            out_shape=(-1, 256, 28, 28),
            padding=(36, 36, 36, 36),
            dilation=(36, 36),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        conv2d_test_util(
            in_shape=(1, 1, 4, 4),
            weight_shape=(2, 1, 2, 2),
            out_shape=(-1, 2, 2, 2),
            padding=(0, 0, 0, 0),
            dilation=(2, 2),
            data_layout="NCHW",
            kernel_layout="OIHW",
        )

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_grouped_conv2d(self):
        conv2d_test_util(
            in_shape=(-1, 8, 3, 3),
            weight_shape=(8, 1, 3, 3),
            out_shape=(-1, 8, 1, 1),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="OIHW",
            groups=8,
        )
        conv2d_test_util(
            in_shape=(-1, 8, 3, 3),
            weight_shape=(4, 2, 3, 3),
            out_shape=(-1, 4, 1, 1),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="OIHW",
            groups=4,
        )

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_conv2d_transpose(self):
        conv2d_test_util(
            in_shape=(-1, 2, 3, 3),
            weight_shape=(2, 4, 3, 3),
            out_shape=(-1, 4, 5, 5),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="OIHW",
            conv_transpose=True
        )
        conv2d_test_util(
            in_shape=(-1, 32, 32, 32),
            weight_shape=(32, 128, 5, 5),
            out_shape=(-1, 128, 36, 36),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="OIHW",
            conv_transpose=True
        )
        conv2d_test_util(
            in_shape=(-1, 32, 128, 1),
            weight_shape=(32, 8, 31, 1),
            out_shape=(-1, 8, 256, 1),
            padding=(14, 0, 15, 0),
            strides=[2, 1],
            data_layout="NCHW",
            kernel_layout="OIHW",
            conv_transpose=True
        )
        # TODO out_padding

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_global_avg_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 5, 5), "float32"))
        avg_pool = relay.nn.global_avg_pool2d(var)

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        X = layers[1]
        assert X.type[0] == "Pooling"
        assert X.bottoms == ["var"]
        assert X.shapes == [-1, 2, 1, 1]
        assert "relay_id" in X.attrs
        assert X.attrs["padding"] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs["insize"] == [5, 5]
        assert X.attrs["outsize"] == [1, 1]
        assert X.attrs["data_layout"] == "NCHW"
        assert X.attrs["strides"] == [1, 1]
        assert X.attrs["kernel_size"] == [5, 5]
        assert X.attrs["pool_type"] == "Avg"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_global_max_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 5, 5), "float32"))
        avg_pool = relay.nn.global_max_pool2d(var)

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        X = layers[1]
        assert X.type[0] == "Pooling"
        assert X.bottoms == ["var"]
        assert X.shapes == [-1, 2, 1, 1]
        assert "relay_id" in X.attrs
        assert X.attrs["padding"] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs["insize"] == [5, 5]
        assert X.attrs["outsize"] == [1, 1]
        assert X.attrs["data_layout"] == "NCHW"
        assert X.attrs["strides"] == [1, 1]
        assert X.attrs["kernel_size"] == [5, 5]
        assert X.attrs["pool_type"] == "Max"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_max_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 4, 4), "float32"))
        avg_pool = relay.nn.max_pool2d(
            var, pool_size=(2, 2), strides=(2, 2), padding=(1, 1)
        )

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        X = layers[1]
        assert X.type[0] == "Pooling"
        assert X.bottoms == ["var"]
        assert X.shapes == [-1, 2, 3, 3]
        assert "relay_id" in X.attrs
        assert X.attrs["padding"] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs["insize"] == [4, 4]
        assert X.attrs["outsize"] == [3, 3]
        assert X.attrs["data_layout"] == "NCHW"
        assert X.attrs["strides"] == [2, 2]
        assert X.attrs["kernel_size"] == [2, 2]
        assert X.attrs["pool_type"] == "Max"

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_padding(self):
        var = relay.var("var", relay.TensorType((1, 2, 4, 4), "float32"))
        pad = relay.nn.pad(var, ((0, 0), (0, 0), (0, 1), (0, 1)))

        func = relay.Function([var], pad)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == "Input"
        assert "relay_id" in layers[0].attrs

        X = layers[1]
        assert X.type[0] == "Pad"
        assert X.bottoms == ["var"]
        assert X.shapes == [-1, 2, 5, 5]
        assert "relay_id" in X.attrs
        assert X.attrs["padding"] == [[0, 0], [0, 0], [0, 1], [0, 1]]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_nn_upsampling(self):
        data = relay.var("data", relay.TensorType((-1, 4, 2, 2), "float32"))

        net = relay.nn.upsampling(data, scale_h=3, scale_w=2)

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        params = {}

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == "Input"
        assert layers[1].type[0] == "Upsampling2D"
        assert "relay_id" in layers[1].attrs
        assert layers[1].shapes == [-1, 4, 6, 4]
        assert layers[1].attrs["scale_h"] == 3
        assert layers[1].attrs["scale_w"] == 2
        assert layers[1].attrs["data_layout"] == "NCHW"
        assert layers[1].attrs["method"] == "nearest_neighbor"
        assert layers[1].attrs["align_corners"] is False
