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
    xcompiler_upsample_nhwc_test,
    xcompiler_resnetv1_block_test,
    xcompiler_conv2d_bias_add_relu_nhwc_oihw_test,
    partition_pad_conv2d_pool2d_nhwc_oihw_test,
)

try:
    import tensorflow as tf

    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestDPUCAHX8H(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    target_registry = TargetRegistry()
    rt_manager = RtManager()

    @classmethod
    def setUpClass(cls):
        import pyxir.contrib.target.DPUCAHX8H

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestDPUCAHX8H.target_registry.unregister_target("DPUCAHX8H-u50")
        TestDPUCAHX8H.target_registry.unregister_target("DPUCAHX8H-u50lv")
        TestDPUCAHX8H.target_registry.unregister_target("DPUCAHX8H-u50lv_dwc")
        TestDPUCAHX8H.target_registry.unregister_target("DPUCAHX8H-u55c_dwc")
        TestDPUCAHX8H.target_registry.unregister_target("DPUCAHX8H-u280")


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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc", 
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
            ],
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
            targets=[
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
            ],
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
            targets=[
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
            ],
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
            targets=["DPUCAHX8H-u55c_dwc", "DPUCAHX8H-u50lv_dwc"],
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
            targets=["DPUCAHX8H-u50", "DPUCAHX8H-u50lv", "DPUCAHX8H-u50lv_dwc", "DPUCAHX8H-u280", "DPUCAHX8H-u55c_dwc",],
        )
    
    def test_upsample(self):
        xcompiler_upsample_nhwc_test(
            in_shape=(1, 112, 112, 64),
            pool_size=[3, 3],
            pool_strides=[2, 2],
            w1_shape=(256, 64, 1, 1),
            scale_h=2,
            scale_w=2,
            data_layout="NHWC",
            method="nearest_neighbor",
            targets=[
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
            ],
         
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
                "DPUCAHX8H-u50",
                "DPUCAHX8H-u50lv",
                "DPUCAHX8H-u50lv_dwc",
                "DPUCAHX8H-u55c_dwc",
                "DPUCAHX8H-u280"
            ],
        )
