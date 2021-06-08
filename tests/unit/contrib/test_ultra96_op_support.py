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

"""Module for testing the dpuv2-ultra96 supported operations"""

import os
import unittest
import numpy as np

from pyxir.graph.layer.xlayer import XLayer
from pyxir.target_registry import TargetRegistry


class TestUltra96OpSupport(unittest.TestCase):

    target_registry = TargetRegistry()

    @classmethod
    def setUpClass(cls):
        def test():
            raise NotImplementedError("")

        TestUltra96OpSupport.target_registry.register_target(
            "dpuv2-ultra96", {}, test, test, test, test
        )

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestUltra96OpSupport.target_registry.unregister_target("dpuv2-ultra96")
        # TestUltra96OpSupport.target_registry.unregister_target('DPUCZDX8G-ultra96')

    def test_batchnorm_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import batchnorm_op_support

        X = XLayer(
            type=["BatchNorm"],
            name="bn1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert batchnorm_op_support(X, [], [])

        X = XLayer(
            type=["BatchNorm"],
            name="bn1",
            shapes=[-1, 2570, 4, 4],
            sizes=[2570 * 16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert not batchnorm_op_support(X, [], [])

    def test_biasadd_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import biasadd_op_support

        X = XLayer(
            type=["BiasAdd"],
            name="bn1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert biasadd_op_support(X, [], [])

        X = XLayer(
            type=["BiasAdd"],
            name="bn1",
            shapes=[-1, 2570, 4, 4],
            sizes=[2570 * 16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert not biasadd_op_support(X, [], [])

    def test_concat_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import concat_op_support

        X = XLayer(
            type=["Concat"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert concat_op_support(X, [], [])

        X = XLayer(
            type=["Concat"],
            name="layer1",
            shapes=[-1, 2570, 4, 4],
            sizes=[2570 * 16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert not concat_op_support(X, [], [])

    def test_conv2d_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import conv2d_op_support

        X = XLayer(
            type=["Convolution"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "dilation": [1, 1],
                "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
                "channels": [4, 2],
                "groups": 1,
            },
        )

        assert conv2d_op_support(X, [], [])

        X = XLayer(
            type=["Convolution"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "dilation": [1, 1],
                "padding": [[0, 0], [0, 0], [3, 3], [1, 1]],
                "channels": [4, 2],
                "groups": 1,
            },
        )

        assert not conv2d_op_support(X, [], [])

    def test_conv2d_transpose_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import conv2d_transpose_op_support

        X = XLayer(
            type=["Conv2DTranspose"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "dilation": [1, 1],
                "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
                "channels": [4, 2],
                "groups": 1,
            },
        )

        assert conv2d_transpose_op_support(X, [], [])

        X = XLayer(
            type=["Conv2DTranspose"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "dilation": [1, 1],
                "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
                "channels": [2570, 2],
                "groups": 1,
            },
        )

        assert not conv2d_transpose_op_support(X, [], [])

    def test_dpuv2_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import dpu_op_support

        X = XLayer(
            type=["DPU"],
            name="layer1",
            shapes=[[-1, 2, 4, 4], [-1, 1, 4, 4]],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={},
        )

        assert dpu_op_support(X, [], [])

    def test_eltwise_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import eltwise_op_support

        X = XLayer(
            type=["Eltwise"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={},
        )

        assert eltwise_op_support(X, [], [])

    def test_pad_pooling_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import pad_op_support

        X = XLayer(
            type=["Pad"],
            name="pad1",
            shapes=[-1, 2, 6, 6],
            sizes=[72],
            bottoms=[],
            tops=["layer1"],
            targets=[],
            attrs={"padding": [[0, 0], [0, 0], [2, 2], [2, 2]]},
        )

        tX = XLayer(
            type=["Pooling"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=["pad1"],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [3, 3],
                "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            },
        )

        assert pad_op_support(X, [], [tX])

        X = XLayer(
            type=["Pad"],
            name="pad1",
            shapes=[-1, 2, 6, 6],
            sizes=[72],
            bottoms=[],
            tops=["layer1"],
            targets=[],
            attrs={"padding": [[0, 0], [0, 0], [5, 2], [5, 2]]},
        )

        tX = XLayer(
            type=["Pooling"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=["pad1"],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [3, 3],
                "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
            },
        )

        assert not pad_op_support(X, [], [tX])

    def test_pad_convolution_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import pad_op_support

        X = XLayer(
            type=["Pad"],
            name="pad1",
            shapes=[-1, 2, 6, 6],
            sizes=[72],
            bottoms=[],
            tops=["layer1"],
            targets=[],
            attrs={"padding": [[0, 0], [0, 0], [1, 1], [1, 1]]},
        )

        tX = XLayer(
            type=["Convolution"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=["pad1"],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "dilation": [1, 1],
                "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
                "channels": [4, 2],
                "groups": 1,
            },
        )

        assert pad_op_support(X, [], [tX])

        X = XLayer(
            type=["Pad"],
            name="pad1",
            shapes=[-1, 2, 6, 6],
            sizes=[72],
            bottoms=[],
            tops=["layer1"],
            targets=[],
            attrs={"padding": [[0, 0], [0, 0], [2, 2], [2, 2]]},
        )

        tX = XLayer(
            type=["Convolution"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=["pad1"],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "dilation": [1, 1],
                "padding": [[0, 0], [0, 0], [0, 0], [0, 0]],
                "channels": [4, 2],
                "groups": 1,
            },
        )

        assert not pad_op_support(X, [], [tX])

    def test_pooling_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import pooling_op_support

        X = XLayer(
            type=["Pooling"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [3, 3],
                "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
            },
        )

        assert pooling_op_support(X, [], [])

        X = XLayer(
            type=["Pooling"],
            name="layer1",
            shapes=[-1, 2570, 4, 4],
            sizes=[2570 * 16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={
                "data_layout": "NCHW",
                "kernel_size": [2, 2],
                "strides": [1, 1],
                "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
            },
        )

        assert not pooling_op_support(X, [], [])

    def test_mean_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import mean_op_support

        X = XLayer(
            type=["Mean"],
            name="layer1",
            shapes=[-1, 2, 1, 1],
            sizes=[2],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axes": [2, 3], "keepdims": True, "exclude": False},
        )

        assert mean_op_support(X, [], [])

        X = XLayer(
            type=["Mean"],
            name="layer1",
            shapes=[-1, 1, 4, 4],
            sizes=[16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axes": [1], "keepdims": True, "exclude": False},
        )

        assert not mean_op_support(X, [], [])

    def test_mean_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import prelu_op_support

        X = XLayer(
            type=["pReLU"],
            name="layer1",
            shapes=[-1, 2, 1, 1],
            sizes=[2],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"alpha": 0.1},
        )

        assert prelu_op_support(X, [], [])

        X = XLayer(
            type=["pReLU"],
            name="layer1",
            shapes=[-1, 1, 4, 4],
            sizes=[16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"alpha": 0.2},
        )

        assert not prelu_op_support(X, [], [])

    def test_relu_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import relu_op_support

        X = XLayer(
            type=["ReLU"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={},
        )

        assert relu_op_support(X, [], [])

    def test_relu6_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import relu6_op_support

        X = XLayer(
            type=["ReLU6"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={},
        )

        assert relu6_op_support(X, [], [])

    def test_scale_support(self):
        from pyxir.contrib.dpuv2.ultra96_op_support import scale_op_support

        X = XLayer(
            type=["Scale"],
            name="layer1",
            shapes=[-1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert scale_op_support(X, [], [])

        X = XLayer(
            type=["Scale"],
            name="layer1",
            shapes=[-1, 2570, 4, 4],
            sizes=[2570 * 16],
            bottoms=[],
            tops=[],
            targets=[],
            attrs={"axis": 1},
        )

        assert not scale_op_support(X, [], [])
