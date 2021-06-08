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

"""Module for testing the decent quantization flow"""

import os
import importlib
import unittest
import numpy as np
import pyxir as px

from pyxir.target_registry import TargetRegistry, register_op_support_check
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.quantization.decent_quantizer import is_decent_available, DECENTQuantizer

from .decent_quantization_infra import conv2d_pool2d_nhwc_oihw_test

if not is_decent_available():
    raise unittest.SkipTest("Skipping DECENT quantization because DECENT is not available")

import logging

logging.basicConfig()
logger = logging.getLogger("pyxir")
# logger.setLevel(logging.DEBUG)


class TestDecentQuantizer(unittest.TestCase):

    xgraph_factory = XGraphFactory()

    @classmethod
    def setUpClass(cls):
        def xgraph_build_func(xgraph):
            raise NotImplementedError("")

        def xgraph_optimizer(xgraph):
            raise NotImplementedError("")

        def xgraph_quantizer(xgraph):
            raise NotImplementedError("")

        def xgraph_compiler(xgraph):
            raise NotImplementedError("")

        target_registry = TargetRegistry()
        target_registry.register_target(
            "test-DPU",
            xgraph_optimizer,
            xgraph_quantizer,
            xgraph_compiler,
            xgraph_build_func,
        )

        @register_op_support_check("test-DPU", "Convolution")
        def conv_op_support(X, bXs, tXs):
            data_layout = X.attrs['data_layout']
            kernel_h, kernel_w = X.attrs['kernel_size']
            stride_h, stride_w = X.attrs['strides']
            dilation_h, dilation_w = X.attrs['dilation']
            padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
                X.attrs['padding'][data_layout.index('W')]
            padding_h_top, padding_h_bot = padding_h
            padding_w_left, padding_w_right = padding_w
            ch_in, ch_out = X.attrs['channels']
            groups = X.attrs['groups']

            return kernel_h >= 1 and kernel_h <= 16 and\
                kernel_w >= 1 and kernel_w <= 16 and\
                stride_h >= 1 and stride_h <= 4 and\
                stride_w >= 1 and stride_w <= 4 and\
                padding_h_top >= 0 and padding_h_top <= kernel_h - 1 and\
                padding_h_bot >= 0 and padding_h_bot <= kernel_h - 1 and\
                padding_w_left >= 0 and padding_w_left <= kernel_w - 1 and\
                padding_w_right >= 0 and padding_w_right <= kernel_w - 1 and\
                ch_in >= 1 and ch_in <= 4096 and\
                ch_out >= 1 and ch_out <= 4096 and\
                dilation_h * ch_in <= 4096 and\
                (dilation_h == 1 or stride_h == 1) and\
                dilation_w * ch_in <= 4096 and\
                (dilation_w == 1 or stride_w == 1)

        @register_op_support_check("test-DPU", "Pooling")
        def pooling_op_support(X, bXs, tXs):
            data_layout = X.attrs['data_layout']

            kernel_h, kernel_w = X.attrs['kernel_size']
            stride_h, stride_w = X.attrs['strides']
            padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
                X.attrs['padding'][data_layout.index('W')]
            padding_h_top, padding_h_bot = padding_h
            padding_w_left, padding_w_right = padding_w

            channels = X.shapes[data_layout.index('C')]

            return kernel_h >= 1 and kernel_h <= 8 and\
                kernel_w >= 1 and kernel_w <= 8 and\
                stride_h >= 1 and stride_h <= 4 and\
                stride_w >= 1 and stride_w <= 4 and\
                padding_h_top >= 0 and padding_h_top <= 4 and\
                padding_h_bot >= 0 and padding_h_bot <= 4 and\
                padding_w_left >= 0 and padding_w_left <= 4 and\
                padding_w_right >= 0 and padding_w_right <= 4 and\
                channels >= 1 and channels <= 4096

        # @register_op_support_check("test", "Concat")
        # def concat_op_support(X, bXs, tXs):
        #     return False

        # @register_op_support_check("test", "Eltwise")
        # def eltwise_op_support(X, bXs, tXs):
        #     return True

        # @register_op_support_check("test", "ReLU")
        # def relu_op_support(X, bXs, tXs):
        #     return True

    @classmethod
    def tearDownClass(cls):

        target_registry = TargetRegistry()
        target_registry.unregister_target("test-DPU")

    # @classmethod
    # def tearDownClass(cls):
    #     # Unregister dpu for other tests
    #     TestDecentQuantizer.target_registry.unregister_target('DPUCZDX8G-zcu102')
    #     TestDecentQuantizer.target_registry.unregister_target('DPUCZDX8G-zcu104')
    #     TestDecentQuantizer.target_registry.unregister_target('DPUCZDX8G-ultra96')
    #     TestDecentQuantizer.target_registry.unregister_target('DPUCZDX8G-som')
    
    def test_pass_conv2d_pool2d_small(self):
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1), (2, 1, 2, 2), [0, 0], [1, 1], [1, 1], "Max", [2, 2], [0, 0],
        )
        # Strided
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1), (2, 1, 2, 2), [0, 0], [2, 2], [1, 1], "Max", [2, 2], [0, 0],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1), (2, 1, 2, 2), [0, 0], [3, 3], [1, 1], "Avg", [2, 2], [1, 1],
        )
        # Padded
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1), (2, 1, 2, 2), [1, 1], [1, 1], [1, 1], "Max", [4, 4], [0, 0],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 8, 8, 1), (2, 1, 3, 3), [2, 2], [1, 1], [1, 1], "Avg", [4, 4], [0, 0],
        )
        # Dilated
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1), (2, 1, 2, 2), [1, 1], [1, 1], [2, 2], "Max", [2, 2], [0, 0],
        )
        conv2d_pool2d_nhwc_oihw_test(
            (1, 10, 10, 1), (2, 1, 2, 2), [1, 1], [1, 1], [4, 4], "Max", [2, 2], [0, 0],
        )

    def test_pass_conv2d_pool2d_large(self):
        # Dilated
        conv2d_pool2d_nhwc_oihw_test(
            (1, 28, 28, 512), (512, 512, 3, 3), [2, 2, 2, 2], [1, 1], [2, 2], "Max", [2, 2], [0, 0],
        )

    def test_pass_depthwise_conv2d_pool2d(self):
        conv2d_pool2d_nhwc_oihw_test(
            (1, 3, 3, 8), (8, 1, 3, 3), [0, 0], [1, 1], [1, 1], "Max", [1, 1], [0, 0], conv_groups=8
        )
        # conv2d_pool2d_nhwc_oihw_test(
        #     (1, 3, 3, 8), (8, 2, 3, 3), [0, 0], [1, 1], [1, 1], "Max", [1, 1], [0, 0], conv_groups=8
        # )

    def test_conv2d_invalid_pool2d_valid(self):
        conv2d_pool2d_nhwc_oihw_test(
            (1, 4, 4, 1), (2, 1, 2, 2), [3, 3, 3, 3], [1, 1], [1, 1], "Max", [2, 2], [0, 0], conv_invalid=True
        ) # Padding values are too large so conv is offloaded to CPU
        
if __name__ == "__main__":
    unittest.main()
