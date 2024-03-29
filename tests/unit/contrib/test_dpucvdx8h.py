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
import numpy as np

from pyxir import partition
from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.target_registry import TargetRegistry
from pyxir.runtime.rt_manager import RtManager

from .compilation_infra import (
    xcompiler_conv2d_pool2d_nhwc_oihw_test,
    xcompiler_scale_conv2d_nhwc_oihw_test,
    xcompiler_conv2d_dropout_pool2d_nhwc_oihw_test,
    xcompiler_conv2d_pool2d_dropout_nhwc_oihw_test,
    xcompiler_resnetv1_block_test,
    xcompiler_conv2d_bias_add_relu_nhwc_oihw_test,
    partition_pad_conv2d_pool2d_nhwc_oihw_test,
)

try:
    import tensorflow as tf

    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestDPUCVDX8H(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    target_registry = TargetRegistry()
    rt_manager = RtManager()

    @classmethod
    def setUpClass(cls):
        import pyxir.contrib.target.DPUCVDX8H

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestDPUCVDX8H.target_registry.unregister_target("DPUCVDX8H")
        TestDPUCVDX8H.target_registry.unregister_target("DPUCVDX8H-dwc")

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
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
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
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
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
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
        )
        # Padded
        # xcompiler_conv2d_pool2d_nhwc_oihw_test(
        #     (1, 4, 4, 1), (2, 1, 2, 2), [1, 1], [1, 1], [1, 1], "Max", [4, 4], [0, 0], targets=["DPUCVDX8H"]
        # )
        # xcompiler_conv2d_pool2d_nhwc_oihw_test(
        #     (1, 8, 8, 1), (2, 1, 3, 3), [2, 2], [1, 1], [1, 1], "Avg", [4, 4], [0, 0], targets=["DPUCVDX8H"]
        # )
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
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
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
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
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
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
        )

    def test_compile_dropout(self):
        xcompiler_conv2d_dropout_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [3, 3],
            [1, 1],
            "Avg",
            [2, 2],
            [1, 1],
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
        )

        xcompiler_conv2d_pool2d_dropout_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [3, 3],
            [1, 1],
            "Avg",
            [2, 2],
            [1, 1],
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
        )

    def test_compile_depthwise_conv2d_pool2d(self):
        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 3, 3, 8),
            (8, 1, 3, 3),
            [0, 0],
            [1, 1],
            [1, 1],
            "Max",
            [1, 1],
            [0, 0],
            conv_groups=8,
            expected_nb_subgraphs=2,
            targets=["DPUCVDX8H"],
        )

        xcompiler_conv2d_pool2d_nhwc_oihw_test(
            (1, 3, 3, 8),
            (8, 1, 3, 3),
            [0, 0],
            [1, 1],
            [1, 1],
            "Max",
            [1, 1],
            [0, 0],
            conv_groups=8,
            targets=["DPUCVDX8H-dwc"],
        )

    def test_compile_scale_conv2d(self):
        # Standalone scale/batchnorm unsupported in DPUCAHX8H compiler
        xcompiler_scale_conv2d_nhwc_oihw_test(
            (1, 299, 299, 3),
            (64, 3, 7, 7),
            [3, 3],
            [2, 2],
            [1, 1],
            expected_nb_subgraphs=5,
            target="DPUCVDX8H",
        )
        # xcompiler_scale_conv2d_nhwc_oihw_test(
        #     (1, 28, 28, 512), (512, 512, 3, 3), [2, 2, 2, 2], [1, 1], [2, 2],
        # )
    @unittest.skip
    def test_compile_conv2d_bias_add_relu(self):
        xcompiler_conv2d_bias_add_relu_nhwc_oihw_test(
            (1, 4, 4, 1),
            (2, 1, 2, 2),
            [0, 0],
            [1, 1],
            [1, 1],
            targets=["DPUCVDX8H", "DPUCVDX8H-dwc"],
        )

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
            target="DPUCVDX8H",
        )

        xcompiler_resnetv1_block_test(
            in_shape=(1, 112, 112, 64),
            pool_size=[3, 3],
            pool_strides=[2, 2],
            w1_shape=(256, 64, 1, 1),
            w2_shape=(64, 64, 1, 1),
            w3_shape=(64, 64, 3, 3),
            w4_shape=(256, 64, 1, 1),
            c3_padding=[1, 1, 1, 1],
            target="DPUCVDX8H-dwc",
        )
    
    def test_pad_conv2d_pool2d_partition(self):
        partition_pad_conv2d_pool2d_nhwc_oihw_test(
            (1, 10, 10, 1),
            [[0, 0], [2, 2], [2, 2], [0, 0]],
            1,
            (2, 1, 2, 2),
            [1, 1],
            [1, 1],
            [4, 4],
            "Max",
            [2, 2],
            [0, 0],
            targets=[
                "DPUCVDX8H",
                "DPUCVDX8H-dwc",
            ],
        )
