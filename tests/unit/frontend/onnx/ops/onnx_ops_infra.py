# Copyright 2021 Xilinx Inc.
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

"""Utilities for testing ONNX to XGraph frontend"""

import onnx
import numpy as np
import pyxir as px

from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l2_convolution as ol2c


def pool_test(
    in_shape,
    pool_type,
    kernel_size,
    padding,
    strides,
    expected_out_shape,
    expected_padding,
) -> bool:

    x = np.random.rand(*in_shape).astype(np.float32)
    if isinstance(padding, str):
        node = onnx.helper.make_node(
            pool_type,
            inputs=["x"],
            outputs=["y"],
            kernel_shape=kernel_size,
            auto_pad=padding,
            strides=strides,
        )
    else:
        node = onnx.helper.make_node(
            pool_type,
            inputs=["x"],
            outputs=["y"],
            kernel_shape=kernel_size,
            pads=padding,
            strides=strides,
        )

    wrapped_node = NodeWrapper(node)

    iX = px.ops.input("x", list(x.shape), dtype="float32")
    if pool_type == "AveragePool":
        Xs = ol2c.avg_pool(wrapped_node, {}, {"x": iX})
        expected_pool_type = "Avg"
    else:
        Xs = ol2c.max_pool(wrapped_node, {}, {"x": iX})
        expected_pool_type = "Max"

    assert len(Xs) == 1
    X = Xs[0]

    assert X.name == "y"
    assert "Pooling" in X.type
    assert X.shapes.tolist() == expected_out_shape
    assert X.attrs["padding"] == expected_padding
    assert X.attrs["strides"] == strides
    assert X.attrs["kernel_size"] == kernel_size
    assert X.attrs["data_layout"] == "NCHW"
    assert X.attrs["type"] == expected_pool_type


def conv_test(
    conv_type,
    in_shape,
    w_shape,
    padding,
    strides,
    dilations,
    groups,
    expected_out_shape,
    expected_padding,
    conv_transpose_out_shape=None,
) -> bool:

    x = np.random.rand(*in_shape).astype(np.float32)
    W = np.random.rand(*w_shape).astype(np.float32)
    B = np.random.rand(w_shape[0]).astype(np.float32)
    kernel_shape = w_shape[2], w_shape[3]
    out_ch, in_ch = w_shape[0], w_shape[1] * groups

    if conv_transpose_out_shape is not None:
        assert conv_type == "ConvTranspose"
        node = onnx.helper.make_node(
            conv_type,
            inputs=["x", "W", "B"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            strides=strides,
            dilations=dilations,
            group=groups,
            output_shape=conv_transpose_out_shape,
        )
    elif isinstance(padding, str):
        node = onnx.helper.make_node(
            conv_type,
            inputs=["x", "W", "B"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            auto_pad=padding,
            strides=strides,
            dilations=dilations,
            group=groups,
        )
    else:
        node = onnx.helper.make_node(
            conv_type,
            inputs=["x", "W", "B"],
            outputs=["y"],
            kernel_shape=kernel_shape,
            pads=padding,
            strides=strides,
            dilations=dilations,
            group=groups,
        )

    wrapped_node = NodeWrapper(node)

    iX = px.ops.input("x", list(x.shape), dtype="float32")
    wX = px.ops.constant("W", W, onnx_id="W")
    bX = px.ops.constant("B", B, onnx_id="B")

    xmap = {"x": iX, "W": wX, "B": bX}
    if conv_type == "Conv":
        Xs = ol2c.conv(wrapped_node, {}, xmap)
    elif conv_type == "ConvTranspose":
        Xs = ol2c.conv_transpose(wrapped_node, {}, xmap)
    else:
        raise ValueError("Unsupported conv type: {0}".format(conv_type))

    assert len(Xs) == 2
    X, baX = Xs

    assert X.name == "y_Conv"
    assert X.shapes.tolist() == expected_out_shape
    assert X.attrs["padding"] == expected_padding
    assert X.attrs["strides"] == strides
    assert X.attrs["dilation"] == dilations
    assert X.attrs["kernel_size"] == kernel_shape
    assert X.attrs["channels"] == [in_ch, out_ch]
    assert X.attrs["data_layout"] == "NCHW"
    assert X.attrs["kernel_layout"] == "OIHW"
    assert X.attrs["groups"] == groups
    assert X.attrs["onnx_id"] == "y"

    assert baX.name == "y"
    assert baX.shapes == expected_out_shape
    assert baX.attrs["axis"] == 1
    assert baX.attrs["onnx_id"] == "y"
