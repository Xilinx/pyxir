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
Module for testing the XOp factory and property functionality


"""

import unittest
import numpy as np

from pyxir.graph import ops
from pyxir.graph.layer.xlayer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf


class TestL5Vision(unittest.TestCase):
    def test_cvx_input_nchw(self):

        iX = XLayer(
            type=["StrInput"],
            name="in1",
            shapes=[-1],
            sizes=[1],
            bottoms=[],
            tops=[],
            targets=[],
        )

        cvx_key = "scale-0.5__transpose-2,0,1"
        cX = xlf.get_xop_factory_func("Cvx")(
            "cvx1", iX, cvx_key, [-1, 3, 10, 10], "float32"
        )

        assert cX.type[0] == "Cvx"
        assert cX.attrs["cvx_key"] == "scale-0.5__transpose-2,0,1"
        assert cX.shapes == [-1, 3, 10, 10]

    def test_yolo_reorg(self):

        iX = XLayer(
            type=["Input"],
            name="in1",
            shapes=[1, 2, 4, 4],
            sizes=[32],
            bottoms=[],
            tops=[],
            targets=[],
        )

        sX = xlf.get_xop_factory_func("YoloReorg")("yr1", iX, 2, "NCHW")

        assert sX.type[0] == "YoloReorg"
        assert sX.shapes == [1, 8, 2, 2]
        assert sX.sizes == [32]
        assert sX.attrs["stride"] == 2
        assert sX.attrs["layout"] == "NCHW"
        assert sX.bottoms == ["in1"]
