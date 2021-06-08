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

"""Module for testing the dpuv1 supported operations"""

import os
import unittest
import numpy as np

from pyxir.graph.layer.xlayer import XLayer
from pyxir.target_registry import TargetRegistry


class TestDpuv1OpSupport(unittest.TestCase):

    target_registry = TargetRegistry()

    @classmethod
    def setUpClass(cls):
        def test():
            raise NotImplementedError("")

        TestDpuv1OpSupport.target_registry.register_target(
            "dpuv1", {}, test, test, test, test
        )

    @classmethod
    def tearDownClass(cls):
        # Unregister dpu for other tests
        TestDpuv1OpSupport.target_registry.unregister_target("dpuv1")
        TestDpuv1OpSupport.target_registry.unregister_target("DPUCADX8G")
