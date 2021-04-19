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

"""Module for testing the XOp factory and property functionality"""

import unittest
import numpy as np
import pyxir as px

from pyxir.graph.layer.xlayer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph import ops

from pyxir.graph.ops.l2_convolution import (
    conv2d_layout_transform,
    pooling_layout_transform,
)


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
    target_kernel_layout="OIHW",
    conv_transpose=False,
):
    iX = px.ops.input("in", shape=list(in_shape))
    kX = px.ops.constant("kernel", np.ones(weight_shape, dtype=np.float32))
    kernel_size = [
        weight_shape[kernel_layout.index("H")],
        weight_shape[kernel_layout.index("W")],
    ]

    if not conv_transpose:
        in_ch = weight_shape[kernel_layout.index("I")] * groups
        channels = weight_shape[kernel_layout.index("O")]
        expected_type = "Convolution"
        X = px.ops.conv2d(
            op_name="conv",
            input_layer=iX,
            weights_layer=kX,
            kernel_size=kernel_size,
            strides=list(strides),
            padding_hw=list(padding),
            dilation=list(dilation),
            groups=groups,
            channels=channels,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            target_kernel_layout=target_kernel_layout,
        )
    else:
        in_ch = weight_shape[kernel_layout.index("I")] * groups
        channels = weight_shape[kernel_layout.index("O")]
        expected_type = "Conv2DTranspose"
        X = px.ops.conv2d_transpose(
            op_name="conv",
            input_layer=iX,
            weights_layer=kX,
            kernel_size=kernel_size,
            strides=list(strides),
            padding_hw=list(padding),
            dilation=list(dilation),
            groups=groups,
            channels=channels,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
            target_kernel_layout=target_kernel_layout,
        )
        # OIHW
        weight_shape = (
            weight_shape[1],
            weight_shape[0],
            weight_shape[2],
            weight_shape[3],
        )

    layout_idx = tuple(["NCHW".index(e) for e in data_layout])
    padding_nchw = [
        [0, 0],
        [0, 0],
        [padding[0], padding[1]],
        [padding[2], padding[3]],
    ]
    padding = [padding_nchw[e] for e in layout_idx]

    assert X.type[0] == expected_type
    assert X.shapes == list(out_shape), "Expected out shape: {0}, but got: {1}".format(
        out_shape, X.shapes
    )
    assert X.attrs["padding"] == padding, "Expected padding: {0}, but got: {1}".format(
        padding, X.attrs["padding"]
    )
    assert X.attrs["data_layout"] == data_layout
    assert X.attrs["kernel_layout"] == target_kernel_layout
    assert X.attrs["shape"] == list(out_shape)
    assert X.attrs["kernel_size"] == kernel_size
    assert X.attrs["strides"] == list(strides)
    assert X.attrs["groups"] == groups
    assert X.attrs["dilation"] == list(dilation)
    assert X.attrs["channels"] == [in_ch, channels]

    np.testing.assert_array_equal(
        X.data.weights, np.ones(weight_shape, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        X.data.biases, np.zeros((channels,), dtype=np.float32),
    )

    conv2d_layout_transform(X, target_layout="NHWC")

    layout_idx = tuple([data_layout.index(e) for e in "NHWC"])
    trans_layout_idx = tuple(["NCHW".index(e) for e in "NHWC"])
    trans_out_shape = [out_shape[e] for e in layout_idx]
    assert X.type[0] == expected_type
    assert X.shapes == trans_out_shape, "Expected out shape: {0}, but got: {1}".format(
        trans_out_shape, X.shapes
    )
    assert X.attrs["data_layout"] == "NHWC"
    padding = [padding_nchw[e] for e in trans_layout_idx]
    assert X.attrs["padding"] == padding, "Expected padding: {0}, but got: {1}".format(
        padding, X.attrs["padding"]
    )


def global_pool2d_test_util(
    in_shape, out_shape, pool_type, data_layout="NCHW",
):
    iX = px.ops.input("in", shape=list(in_shape))
    X = px.ops.global_pool2d(
        op_name="pool", input_layer=iX, pool_type=pool_type, layout=data_layout,
    )
    expected_padding = [[0, 0], [0, 0], [0, 0], [0, 0]]

    layout_idx = tuple(["NCHW".index(e) for e in data_layout])
    B_idx, C_idx, H_idx, W_idx = layout_idx

    assert X.type[0] == "Pooling"
    assert X.shapes == list(out_shape), "Expected out shape: {0}, but got: {1}".format(
        out_shape, X.shapes
    )
    assert (
        X.attrs["padding"] == expected_padding
    ), "Expected padding: {0}, but got: {1}".format(
        expected_padding, X.attrs["padding"]
    )

    assert X.attrs["insize"] == [in_shape[H_idx], in_shape[W_idx]]
    assert X.attrs["outsize"] == [1, 1]
    assert X.attrs["data_layout"] == data_layout
    assert X.attrs["strides"] == [1, 1]
    assert X.attrs["kernel_size"] == [in_shape[H_idx], in_shape[W_idx]]
    assert X.attrs["pool_type"] == pool_type

    # Test layout transform
    pooling_layout_transform(X, target_layout="NHWC")

    layout_idx = tuple([data_layout.index(e) for e in "NHWC"])
    trans_layout_idx = tuple(["NCHW".index(e) for e in "NHWC"])
    trans_out_shape = [out_shape[e] for e in layout_idx]
    assert X.type[0] == "Pooling"
    assert X.shapes == trans_out_shape, "Expected out shape: {0}, but got: {1}".format(
        trans_out_shape, X.shapes
    )
    assert X.attrs["data_layout"] == "NHWC"
    assert (
        X.attrs["padding"] == expected_padding
    ), "Expected padding: {0}, but got: {1}".format(
        expected_padding, X.attrs["padding"]
    )


def pool2d_test_util(
    in_shape,
    out_shape,
    pool_type,
    pool_size,
    padding=(0, 0, 0, 0),
    strides=(1, 1),
    data_layout="NCHW",
):
    iX = px.ops.input("in", shape=list(in_shape))
    X = px.ops.pool2d(
        op_name="pool",
        input_layer=iX,
        pool_type=pool_type,
        pool_size=list(pool_size),
        strides=list(strides),
        padding=list(padding),
        layout=data_layout,
        ceil_mode=True,
        count_include_pad=True,
    )

    layout_idx = tuple(["NCHW".index(e) for e in data_layout])
    B_idx, C_idx, H_idx, W_idx = layout_idx

    padding_nchw = [
        [0, 0],
        [0, 0],
        [padding[0], padding[2]],
        [padding[1], padding[3]],
    ]
    padding = [padding_nchw[e] for e in layout_idx]

    assert X.type[0] == "Pooling"
    assert X.shapes == list(out_shape), "Expected out shape: {0}, but got: {1}".format(
        out_shape, X.shapes
    )
    assert X.attrs["padding"] == padding, "Expected padding: {0}, but got: {1}".format(
        padding, X.attrs["padding"]
    )

    assert X.attrs["insize"] == [in_shape[H_idx], in_shape[W_idx]]
    assert X.attrs["outsize"] == [out_shape[H_idx], out_shape[W_idx]]
    assert X.attrs["data_layout"] == data_layout
    assert X.attrs["strides"] == list(strides)
    assert X.attrs["kernel_size"] == list(pool_size)
    assert X.attrs["pool_type"] == pool_type

    # Test layout transform
    pooling_layout_transform(X, target_layout="NHWC")

    layout_idx = tuple([data_layout.index(e) for e in "NHWC"])
    trans_layout_idx = tuple(["NCHW".index(e) for e in "NHWC"])
    trans_out_shape = [out_shape[e] for e in layout_idx]
    assert X.type[0] == "Pooling"
    assert X.shapes == trans_out_shape, "Expected out shape: {0}, but got: {1}".format(
        trans_out_shape, X.shapes
    )
    assert X.attrs["data_layout"] == "NHWC"
    padding = [padding_nchw[e] for e in trans_layout_idx]
    assert X.attrs["padding"] == padding, "Expected padding: {0}, but got: {1}".format(
        padding, X.attrs["padding"]
    )


class TestL2Convolution(unittest.TestCase):
    def test_nn_batch_flatten_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 1, 1, 4],
            sizes=[4],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = px.ops.batch_flatten("flatten1", [iX])

        assert sX.type[0] == "Flatten"
        assert sX.shapes == [1, 4]
        assert sX.sizes == [4]
        assert sX.attrs == {}

    def test_batchnorm_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        mX = XLayer(
            type=["Constant"],
            name="mu",
            shapes=[2],
            sizes=[2],
            data=[np.array([0.5, 1.0])],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sqX = XLayer(
            type=["Constant"],
            name="sigma_square",
            shapes=[2],
            sizes=[2],
            data=[np.array([1.0, 2.0])],
            bottoms=[],
            tops=[],
            targets=[],
        )

        gX = XLayer(
            type=["Constant"],
            name="gamma",
            shapes=[2],
            sizes=[2],
            data=[np.array([1.0, 2.0])],
            bottoms=[],
            tops=[],
            targets=[],
        )

        bX = XLayer(
            type=["Constant"],
            name="beta",
            shapes=[2],
            sizes=[2],
            data=[np.array([1.0, -2.0])],
            bottoms=[],
            tops=[],
            targets=[],
        )

        bX = px.ops.batch_norm("bn1", iX, mX, sqX, gX, bX, axis=1, epsilon=1e-5)

        assert bX.type[0] == "BatchNorm"
        assert bX.attrs["axis"] == 1
        assert bX.attrs["epsilon"] == 1e-5

        np.testing.assert_array_equal(bX.data.gamma, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(bX.data.beta, np.array([1.0, -2.0]))
        np.testing.assert_array_equal(bX.data.mu, np.array([0.5, 1.0]))
        np.testing.assert_array_equal(bX.data.sigma_square, np.array([1.0, 2.0]))

        from pyxir.graph.ops.l2_convolution import batchnorm_transpose_transform

        batchnorm_transpose_transform(bX, axes=[0, 2, 3, 1])

        assert bX.type[0] == "BatchNorm"
        assert bX.shapes == [1, 4, 4, 2]
        assert bX.attrs["axis"] == 3
        assert bX.attrs["epsilon"] == 1e-5

    def test_convolution_layer(self):
        conv2d_test_util(
            (1, 2, 3, 3),
            (4, 2, 3, 3),
            (-1, 4, 3, 3),
            padding=(1, 1, 1, 1),
            strides=(1, 1),
            dilation=(1, 1),
            groups=1,
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        conv2d_test_util(
            (1, 3, 3, 2),
            (4, 2, 3, 3),
            (-1, 3, 3, 4),
            padding=(1, 1, 1, 1),
            strides=(1, 1),
            dilation=(1, 1),
            groups=1,
            data_layout="NHWC",
            kernel_layout="OIHW",
        )

    def test_convolution_layer_tfl(self):
        conv2d_test_util(
            (1, 3, 3, 2),
            (4, 3, 3, 2),
            (-1, 3, 3, 4),
            padding=(1, 1, 1, 1),
            strides=(1, 1),
            dilation=(1, 1),
            groups=1,
            data_layout="NHWC",
            kernel_layout="OHWI",
            target_kernel_layout="OHWI",
        )

    def test_depthwise_convolution_layer(self):
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

    def test_conv2d_transpose_layer(self):
        conv2d_test_util(
            in_shape=(-1, 2, 3, 3),
            weight_shape=(2, 4, 3, 3),
            out_shape=(-1, 4, 5, 5),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="IOHW",
            conv_transpose=True,
        )
        conv2d_test_util(
            in_shape=(-1, 32, 32, 32),
            weight_shape=(32, 128, 5, 5),
            out_shape=(-1, 128, 36, 36),
            padding=(0, 0, 0, 0),
            data_layout="NCHW",
            kernel_layout="IOHW",
            conv_transpose=True,
        )
        conv2d_test_util(
            in_shape=(-1, 32, 128, 1),
            weight_shape=(32, 8, 31, 1),
            out_shape=(-1, 8, 256, 1),
            padding=(14, 15, 0, 0),
            strides=[2, 1],
            data_layout="NCHW",
            kernel_layout="IOHW",
            conv_transpose=True,
        )

    def test_global_pooling_layer(self):
        global_pool2d_test_util(
            in_shape=(1, 2, 7, 7), out_shape=(-1, 2, 1, 1), pool_type="Max",
        )
        global_pool2d_test_util(
            in_shape=(1, 2, 8, 8), out_shape=(-1, 2, 1, 1), pool_type="Avg",
        )

    def test_pad_layer(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 7, 7],
            sizes=[98],
            bottoms=[],
            tops=[],
            targets=[],
        )

        X = xlf.get_xop_factory_func("Pad")(
            op_name="pad1",
            padding=[[0, 0], [0, 0], [1, 0], [1, 0]],
            pad_value=0,
            input_layer=iX,
        )

        assert X.type[0] == "Pad"
        assert X.shapes == [1, 2, 8, 8]
        assert X.sizes == [128]
        assert X.attrs["padding"] == [[0, 0], [0, 0], [1, 0], [1, 0]]

        from pyxir.graph.ops.l2_convolution import padding_transpose_transform

        padding_transpose_transform(X, axes=(0, 2, 3, 1))

        assert X.type[0] == "Pad"
        assert X.shapes == [1, 8, 8, 2]
        assert X.attrs["padding"] == [[0, 0], [1, 0], [1, 0], [0, 0]]

    def test_pooling_layer(self):
        pool2d_test_util(
            in_shape=(1, 2, 5, 5),
            out_shape=(-1, 2, 3, 3),
            pool_type="Avg",
            pool_size=[3, 3],
            padding=[1, 1, 1, 1],
            strides=[2, 2],
        )
        pool2d_test_util(
            in_shape=(1, 2, 6, 6),
            out_shape=(-1, 2, 7, 7),
            pool_type="Max",
            pool_size=[3, 3],
            padding=[2, 2, 1, 1],
            strides=[1, 1],
        )

    def test_nn_upsampling2d(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 4, 2, 2],
            sizes=[16],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = xlf.get_xop_factory_func("Upsampling2D")(
            "ups1",
            [iX],
            scale_h=3,
            scale_w=2,
            data_layout="NCHW",
            method="nearest_neighbor",
            align_corners=False,
        )

        assert sX.type[0] == "Upsampling2D"
        assert sX.shapes == [1, 4, 6, 4]
        assert sX.sizes == [96]
        assert sX.attrs["scale_h"] == 3
        assert sX.attrs["scale_w"] == 2
        assert sX.attrs["data_layout"] == "NCHW"
        assert sX.attrs["method"] == "nearest_neighbor"
        assert sX.attrs["align_corners"] is False

        from pyxir.graph.ops.l2_convolution import upsampling2d_layout_transform

        upsampling2d_layout_transform(sX, target_layout="NHWC")

        assert sX.type[0] == "Upsampling2D"
        assert sX.shapes == [1, 6, 4, 4]
        assert sX.attrs["data_layout"] == "NHWC"
