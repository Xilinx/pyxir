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
Module for testing the xgraph XLayer data type


"""

import unittest
import numpy as np
import libpyxir as lpx

from pyxir.graph.layer.xlayer import XLayer, ConvData, defaultXLayer


class TestXLayerFactory(unittest.TestCase):

    def test_xlayer_factory(self):

        X1 = defaultXLayer()

        assert X1.layer == lpx.StrVector([])
        assert X1.tops == lpx.StrVector([])
        assert X1.bottoms == lpx.StrVector([])
        assert X1.targets == lpx.StrVector([])
        assert X1.target == 'cpu'

        X1 = X1._replace(
            name='test',
            tops=['test']
        )
        assert X1.name == 'test'
        assert X1.tops == lpx.StrVector(['test'])

        X2 = defaultXLayer()
        assert X2.layer == lpx.StrVector([])
        assert X2.tops == lpx.StrVector([])
        assert X2.bottoms == lpx.StrVector([])
        assert X2.targets == lpx.StrVector([])
