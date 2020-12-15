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

"""Module for testing the ONNX 2 XLayer registry"""

import unittest

from pyxir.frontend.onnx.onnx_2_xlayer_registry import ONNX2XLayerRegistry,\
    register_onnx_2_xlayer_converter

class MockONNXNode(object):

    def get_inputs(self):
        return []

class TestONNX2XLayerRegistry(unittest.TestCase):

    registry = ONNX2XLayerRegistry()

    def test_register_onnx_2_xlayer_converter(self):

        @register_onnx_2_xlayer_converter("TestOp")
        def converter(node, params, xmap):
            return []

        assert 'TestOp' in TestONNX2XLayerRegistry.registry

        res = TestONNX2XLayerRegistry.registry['TestOp'](MockONNXNode(), {}, {})

        assert res == []
