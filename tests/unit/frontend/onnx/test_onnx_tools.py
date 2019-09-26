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

"""
Module for testing the pyxir ONNX frontend


"""

import onnx
import unittest
import numpy as np

from pyxir.frontend.onnx.onnx_tools import NodeWrapper


class TestNodeWrapper(unittest.TestCase):

    def test_conv_node(self):
        node = onnx.helper.make_node(
            'Conv',
            inputs=['x', 'W'],
            outputs=['y'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        wrapped_node = NodeWrapper(node)

        assert wrapped_node.get_op_type() == 'Conv'
        assert wrapped_node.get_inputs() == ['x', 'W']
        assert wrapped_node.get_outputs() == ['y']

        attrs = wrapped_node.get_attributes()
        assert attrs['kernel_shape'] == [3, 3]
        assert attrs['pads'] == [1, 1, 1, 1]
