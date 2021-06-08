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

"""Module for testing the DPUCZDX8G build functionality"""

import os
import unittest
import logging
import numpy as np

from pyxir import partition
from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.target_registry import TargetRegistry
from pyxir.runtime.rt_manager import RtManager
from pyxir.contrib.target.components.common import is_dpuczdx8g_vart_flow_enabled

from .compilation_infra import (
    xcompiler_conv2d_pool2d_nhwc_oihw_test,
    xcompiler_scale_conv2d_nhwc_oihw_test,
    xcompiler_resnetv1_block_test,
    xcompiler_conv2d_leaky_relu_nhwc_oihw_test,
    conv2d_pool2d_nhwc_oihw_test,
    conv2d_leaky_relu_nhwc_oihw_test,
    conv2d_pool2d_naming_test,
    multi_output_conv2d_naming_test,
)

# logging.basicConfig()
# logger = logging.getLogger("pyxir")
# logger.setLevel(logging.DEBUG)

try:
    import tensorflow as tf

    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestDPUCZDX8G(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    target_registry = TargetRegistry()
    rt_manager = RtManager()

    @classmethod
    def setUpClass(cls):

        # Import DPU module
        from pyxir.contrib.dpuv2 import dpuv2

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestDPUCZDX8G.target_registry.unregister_target("dpuv2-zcu104")
        TestDPUCZDX8G.target_registry.unregister_target("dpuv2-zcu102")
        TestDPUCZDX8G.target_registry.unregister_target("DPUCZDX8G-zcu102")
        TestDPUCZDX8G.target_registry.unregister_target("DPUCZDX8G-zcu104")
        TestDPUCZDX8G.target_registry.unregister_target("dpuv2-ultra96")
        TestDPUCZDX8G.target_registry.unregister_target("DPUCZDX8G-ultra96")
        TestDPUCZDX8G.target_registry.unregister_target("dpuv2-som")
        TestDPUCZDX8G.target_registry.unregister_target("DPUCZDX8G-som")

    # @unittest.skipIf(skip_tf, "Skipping Tensorflow related test because tensorflow is"
    #                 "not available")
    # def test_import_ext_quantizer(self):
    #     if TestDPUCZDX8G.target_registry.is_target('DPUCZDX8G-ultra96'):
    #         TestDPUCZDX8G.target_registry.unregister_target('DPUCZDX8G-ultra96')
    #         TestDPUCZDX8G.target_registry.unregister_target('DPUCZDX8G-zcu104')
    #         TestDPUCZDX8G.target_registry.unregister_target('DPUCZDX8G-zcu102')
    #         TestDPUCZDX8G.target_registry.unregister_target('DPUCZDX8G-som')
    #     if TestDPUCZDX8G.rt_manager.exists_op('cpu-np', 'DPU'):
    #         TestDPUCZDX8G.rt_manager.unregister_op('cpu-np', 'DPU')
    #     from pyxir.contrib.target import DPUCZDX8G_external_quantizer

    @unittest.skipIf(not is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G VART test")
    def test_compile_conv2d_pool2d(self):
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [1, 1],
            [1, 1],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        # Strided
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [2, 2],
            [1, 1],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [3, 3],
            [1, 1],
            "Avg",
            [2, 2],
            [1, 1],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        # Padded
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [1, 1],
            "Max",
            [4, 4],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 8, 8, 1),
            (2, 1, 3, 3),
            [2, 2],
            [1, 1],
            [1, 1],
            "Avg",
            [4, 4],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        # Dilated
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [2, 2],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 10, 10, 1),
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [4, 4],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 28, 28, 512),
            (512, 512, 3, 3),
            [2, 2, 2, 2],
            [1, 1],
            [2, 2],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )

    @unittest.skipIf(is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G DNNC/DNNDK test")
    def test_compile_conv2d_pool2d_dnnc(self):
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [1, 1],
            [1, 1],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        # Strided
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [2, 2],
            [1, 1],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [3, 3],
            [1, 1],
            "Avg",
            [2, 2],
            [1, 1],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        # Padded
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [1, 1],
            "Max",
            [4, 4],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 8, 8, 1),
            (2, 1, 3, 3),
            [2, 2],
            [1, 1],
            [1, 1],
            "Avg",
            [4, 4],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        # Dilated
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [2, 2],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 10, 10, 1),
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [4, 4],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 28, 28, 512),
            (512, 512, 3, 3),
            [2, 2, 2, 2],
            [1, 1],
            [2, 2],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )

    @unittest.skipIf(not is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G VART test")
    def test_compile_depthwise_conv2d_pool2d(self):
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 3, 3, 8),
            (8, 1, 3, 3),
            [1, 1, 1, 1],
            [1, 1],
            [1, 1],
            "Max",
            [2, 2],
            [0, 0],
            conv_groups=8,
            targets=["DPUCZDX8G-zcu104", "DPUCZDX8G-zcu102", "DPUCZDX8G-som"],
            expected_nb_subgraphs=3,
        )

    @unittest.skipIf(not is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G VART test")
    def test_compile_scale_conv2d(self):
        # Standalone scale/batchnorm unsupported in DPUCAHX8H compiler
        xcompiler_scale_conv2d_nhwc_oihw_test(
            (1, 299, 299, 3),
            (64, 3, 7, 7),
            [3, 3],
            [2, 2],
            [1, 1],
            target="DPUCZDX8G-zcu104",
            expected_nb_subgraphs=3,
        )

    @unittest.skipIf(not is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G VART test")
    def test_compile_resnetv1_block(self):
        xcompiler_resnetv1_block_test(
            in_shape=(1, 112, 112, 64),
            pool_size=[3, 3],
            pool_strides=[2, 2],
            w1_shape=(256, 64, 1, 1),
            w2_shape=(64, 64, 1, 1),
            w3_shape=(64, 64, 3, 3),
            w4_shape=(256, 64, 1, 1),
            c3_padding=[1, 1, 1, 1],
            target="DPUCZDX8G-zcu104",
        )

    @unittest.skipIf(is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G DNNC/DNNDK test")
    def test_compile_conv2d_leaky_relu_dnnc(self):
        conv2d_leaky_relu_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [1, 1],
            [1, 1],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )

    @unittest.skipIf(not is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G VART test")
    def test_compile_conv2d_leaky_relu(self):
        xcompiler_conv2d_leaky_relu_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [1, 1],
            [1, 1],
            targets=[
                "DPUCZDX8G-zcu104",
                "DPUCZDX8G-zcu102",
                "DPUCZDX8G-ultra96",
                "DPUCZDX8G-som",
            ],
        )

    @unittest.skipIf(is_dpuczdx8g_vart_flow_enabled(), "DPUCZDX8G DNNC/DNNDK test")
    def test_dnnc_out_names(self):
        multi_output_conv2d_naming_test(["nn.conv-134", "nn.relu-22"])
        multi_output_conv2d_naming_test(["nn..con.v--1.", "n.n.-relu-2-"])
        conv2d_pool2d_naming_test(["conv1", "nn.conv-1"], ["nn.pool1", "nn.pool-1"])

    def test_supported_ops(self):
        ultra96_ops = TestDPUCZDX8G.target_registry.get_supported_op_check_names(
            "dpuv2-ultra96"
        )

        assert "BatchNorm" in ultra96_ops
        assert "BiasAdd" in ultra96_ops
        assert "Concat" in ultra96_ops
        assert "Convolution" in ultra96_ops
        assert "Conv2DTranspose" in ultra96_ops
        assert "DPU" in ultra96_ops
        assert "Eltwise" in ultra96_ops
        assert "Pad" in ultra96_ops
        assert "Pooling" in ultra96_ops
        assert "Mean" in ultra96_ops
        assert "pReLU" in ultra96_ops
        assert "ReLU" in ultra96_ops
        assert "ReLU6" in ultra96_ops
        assert "Scale" in ultra96_ops

        zcu102_ops = TestDPUCZDX8G.target_registry.get_supported_op_check_names(
            "dpuv2-zcu102"
        )

        assert "BatchNorm" in zcu102_ops
        assert "BiasAdd" in zcu102_ops
        assert "Concat" in zcu102_ops
        assert "Convolution" in zcu102_ops
        assert "Conv2DTranspose" in zcu102_ops
        assert "DPU" in zcu102_ops
        assert "Eltwise" in zcu102_ops
        assert "Pad" in zcu102_ops
        assert "Pooling" in zcu102_ops
        assert "Mean" in zcu102_ops
        assert "pReLU" in zcu102_ops
        assert "ReLU" in zcu102_ops
        assert "ReLU6" in zcu102_ops
        assert "Scale" in zcu102_ops

        zcu104_ops = TestDPUCZDX8G.target_registry.get_supported_op_check_names(
            "dpuv2-zcu104"
        )

        assert "BatchNorm" in zcu104_ops
        assert "BiasAdd" in zcu104_ops
        assert "Concat" in zcu104_ops
        assert "Convolution" in zcu104_ops
        assert "Conv2DTranspose" in zcu104_ops
        assert "DPU" in zcu104_ops
        assert "Eltwise" in zcu104_ops
        assert "Pad" in zcu104_ops
        assert "Pooling" in zcu104_ops
        assert "Mean" in zcu104_ops
        assert "pReLU" in zcu104_ops
        assert "ReLU" in zcu104_ops
        assert "ReLU6" in zcu104_ops
        assert "Scale" in zcu104_ops

        som_ops = TestDPUCZDX8G.target_registry.get_supported_op_check_names(
            "dpuv2-som"
        )

        assert "BatchNorm" in som_ops
        assert "BiasAdd" in som_ops
        assert "Concat" in som_ops
        assert "Convolution" in som_ops
        assert "Conv2DTranspose" in som_ops
        assert "DPU" in som_ops
        assert "Eltwise" in som_ops
        assert "Pad" in som_ops
        assert "Pooling" in som_ops
        assert "Mean" in som_ops
        assert "pReLU" in som_ops
        assert "ReLU" in som_ops
        assert "ReLU6" in som_ops
        assert "Scale" in som_ops
        assert "Upsampling2D" in som_ops

    def test_small(self):
        net = [
            XLayer(
                name="in1",
                type=["Input"],
                shapes=[1, 1, 4, 4],
                sizes=[16],
                bottoms=[],
                tops=["conv1"],
                layer=["in1"],
                targets=[],
            ),
            XLayer(
                name="in2",
                type=["Input"],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=[],
                tops=["dense1"],
                layer=["in2"],
                targets=[],
            ),
            XLayer(
                name="conv1",
                type=["Convolution"],
                shapes=[1, 2, 3, 3],
                sizes=[18],
                bottoms=["in1"],
                tops=["pool1"],
                layer=["conv1"],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                attrs={
                    "data_layout": "NCHW",
                    "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
                    "kernel_size": [3, 3],
                    "strides": [1, 1],
                    "dilation": [1, 1],
                    "groups": 1,
                    "channels": [2, 2],
                },
                targets=[],
            ),
            XLayer(
                name="pool1",
                type=["Pooling"],
                shapes=[1, 2, 2, 2],
                sizes=[8],
                bottoms=["conv1"],
                tops=["dense1"],
                layer=["pool1"],
                attrs={
                    "data_layout": "NCHW",
                    "padding": [[0, 0], [0, 0], [1, 1], [1, 1]],
                    "kernel_size": [3, 3],
                    "strides": [1, 1],
                },
                targets=[],
            ),
            XLayer(
                name="dense1",
                type=["Dense"],
                shapes=[1, 20],
                sizes=[20],
                bottoms=["pool1", "in2"],
                tops=[],
                data=ConvData(np.array([1, 1]), np.array([0, 0])),
                layer=["dense1"],
                targets=[],
            ),
        ]
        xgraph = TestDPUCZDX8G.xgraph_factory.build_from_xlayer(net)
        p_xgraph = partition(xgraph, ["dpuv2-zcu104"])
        dpu_xgraph = TestDPUCZDX8G.target_registry.get_target_build_func(
            "dpuv2-zcu104"
        )(p_xgraph)

        assert len(dpu_xgraph) == 6
        layers = dpu_xgraph.get_layers()

        assert layers[0].type[0] == "Input"

        assert layers[1].type[0] == "Transpose"
        assert layers[1].bottoms == ["in1"]
        assert layers[1].tops == ["xp0"]

        assert layers[2].type[0] == "DPU"
        assert layers[2].bottoms == ["conv1_bottom_NCHW>NHWC"]
        assert layers[2].tops == ["pool1"]
        assert layers[2].shapes == [[1, 2, 2, 2]]
        assert layers[2].attrs["target"] == "dpuv2-zcu104"
        assert layers[2].attrs["input_names"] == ["xinput0"]
        assert layers[2].attrs["output_names"] == ["pool1"]
        assert layers[2].attrs["input_layers"]["xinput0"] == ["conv1"]
        assert layers[2].attrs["output_layers"]["pool1"] == ["pool1"]
        assert layers[2].attrs["__top_tensors"] == {"pool1": ["pool1_top_NHWC>NCHW"]}
        assert layers[2].attrs["orig_top_tensors"] == {"pool1": ["dense1"]}
        assert layers[2].attrs["__bottom_tensors"] == {
            "xinput0": ["conv1_bottom_NCHW>NHWC"]
        }
        assert layers[2].attrs["orig_bottom_tensors"] == {"xinput0": ["in1"]}

        # Merged TupleGetItem and Transpose layer
        assert layers[3].type[0] == "TupleGetItem"
        assert layers[3].name == "pool1"
        assert layers[3].shapes == [1, 2, 2, 2]
        assert layers[3].bottoms == ["xp0"]
        assert layers[3].tops == ["dense1"]
        assert layers[3].attrs["transpose"] is True

        assert layers[4].type[0] == "Input"
        assert layers[4].name == "in2"
        assert layers[4].tops == ["dense1"]

        assert layers[5].type[0] == "Dense"
        assert layers[5].name == "dense1"
        assert layers[5].shapes == [1, 20]
        assert layers[5].bottoms == ["pool1", "in2"]
        assert layers[5].tops == []
