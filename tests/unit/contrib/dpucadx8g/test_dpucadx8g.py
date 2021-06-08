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

"""Module for testing the DPUCADX8G target"""

import os
import unittest
import pyxir
import numpy as np

from pyxir.graph.layer.xlayer import XLayer, ConvData
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.target_registry import TargetRegistry
from pyxir.runtime.rt_manager import RtManager

try:
    import tensorflow as tf

    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestDPUContrib(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    target_registry = TargetRegistry()
    rt_manager = RtManager()

    @classmethod
    def setUpClass(cls):
        # Import DPU module
        from pyxir.contrib.dpuv1 import dpuv1

        # from pyxir.contrib.dpuv1.dpuv1_target import\
        #     xgraph_dpu_v1_optimizer,\
        #     xgraph_dpu_v1_quantizer,\
        #     xgraph_dpu_v1_compiler,\
        #     xgraph_dpu_v1_build_func

        # pyxir.register_target(
        #     'dpuv1',
        #     xgraph_dpu_v1_optimizer,
        #     xgraph_dpu_v1_quantizer,
        #     xgraph_dpu_v1_compiler,
        #     xgraph_dpu_v1_build_func
        # )

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestDPUContrib.target_registry.unregister_target("dpuv1")
        TestDPUContrib.target_registry.unregister_target("DPUCADX8G")

    def test_supported_ops(self):
        dpuv1_ops = TestDPUContrib.target_registry.get_supported_op_check_names("dpuv1")

        assert "BatchNorm" in dpuv1_ops
        assert "BiasAdd" in dpuv1_ops
        assert "Concat" in dpuv1_ops
        assert "Convolution" in dpuv1_ops
        assert "Conv2DTranspose" in dpuv1_ops
        assert "DPU" in dpuv1_ops
        assert "Eltwise" in dpuv1_ops
        assert "Pad" in dpuv1_ops
        assert "Pooling" in dpuv1_ops
        assert "Mean" in dpuv1_ops
        assert "pReLU" in dpuv1_ops
        assert "ReLU" in dpuv1_ops
        assert "Scale" in dpuv1_ops

    @unittest.skipIf(
        skip_tf,
        "Skipping Tensorflow related test because tensorflow is" "not available",
    )
    def test_import_ext_quantizer(self):
        if TestDPUContrib.target_registry.is_target("DPUCADX8G"):
            TestDPUContrib.target_registry.unregister_target("DPUCADX8G")
        if TestDPUContrib.rt_manager.exists_op("cpu-np", "DPU"):
            TestDPUContrib.rt_manager.unregister_op("cpu-np", "DPU")
        from pyxir.contrib.target import DPUCADX8G_external_quantizer
