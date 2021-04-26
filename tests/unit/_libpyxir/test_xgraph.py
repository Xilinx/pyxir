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

"""Module for testing the libpyxir XGraph data structure"""

import unittest
import numpy as np
import pyxir._libpyxir as lpx


class TestXGraph(unittest.TestCase):
    def test_constructor(self):
        xg = lpx.XGraph("g")
        assert xg.get_name() == "g"

    def test_name(self):
        xg = lpx.XGraph("g")
        assert xg.get_name() == "g"
        xg.set_name("g2")
        assert xg.get_name() == "g2"

    def test_add(self):
        xg = lpx.XGraph("g")
        X1 = lpx.XLayer(
            name="x1", xtype=lpx.StrVector(["Input"]), bottoms=lpx.StrVector([])
        )

        xg.add(X1)
        assert "x1" in xg
        assert len(xg.get_input_names()) == 1
        assert xg.get_input_names()[0] == "x1"
        assert len(xg.get_output_names()) == 1
        assert xg.get_output_names()[0] == "x1"

        X2 = lpx.XLayer(
            name="x2", xtype=lpx.StrVector(["X2"]), bottoms=lpx.StrVector(["x1"])
        )
        xg.add(X2)

        assert len(xg) == 2

        X1_xg = xg.get("x1")
        assert len(X1_xg.tops) == 1
        assert X1_xg.tops[0] == "x2"
        assert len(xg.get_input_names()) == 1
        assert xg.get_input_names()[0] == "x1"
        assert len(xg.get_output_names()) == 1
        assert xg.get_output_names()[0] == "x2"

        assert xg.get_layer_names() == lpx.StrVector(["x1", "x2"])

        X3 = lpx.XLayer(
            name="x3", xtype=lpx.StrVector(["X3"]), bottoms=lpx.StrVector(["x2"])
        )
        xg.add(X3)
        assert xg.get_input_names() == lpx.StrVector(["x1"])
        assert xg.get_output_names() == lpx.StrVector(["x3"])
        assert xg.get_layer_names() == lpx.StrVector(["x1", "x2", "x3"])
        assert xg.get("x2").tops == lpx.StrVector(["x3"])
        assert xg.get("x2").bottoms == lpx.StrVector(["x1"])

        X4 = lpx.XLayer(
            name="x4",
            xtype=lpx.StrVector(["X4"]),
            bottoms=lpx.StrVector(["x1"]),
            tops=lpx.StrVector(["x3"]),
        )
        xg.add(X4)
        assert xg.get_input_names() == lpx.StrVector(["x1"])
        assert xg.get_output_names() == lpx.StrVector(["x3"])
        assert xg.get_layer_names() == lpx.StrVector(["x1", "x2", "x4", "x3"])
        assert xg.get("x2").tops == lpx.StrVector(["x3"])
        assert xg.get("x2").bottoms == lpx.StrVector(["x1"])
        assert xg.get("x4").tops == lpx.StrVector(["x3"])
        assert xg.get("x4").bottoms == lpx.StrVector(["x1"])

        X5 = lpx.XLayer(
            name="x5",
            xtype=lpx.StrVector(["Input"]),
            bottoms=lpx.StrVector([]),
            tops=lpx.StrVector(["x4"]),
        )
        X6 = lpx.XLayer(
            name="x6",
            xtype=lpx.StrVector(["X6"]),
            bottoms=lpx.StrVector(["x4"]),
            tops=lpx.StrVector([]),
        )
        xg.add(X5)
        xg.add(X6)
        assert xg.get_input_names() == lpx.StrVector(["x1", "x5"])
        assert xg.get_output_names() == lpx.StrVector(["x3", "x6"])
        assert xg.get_layer_names() == lpx.StrVector(
            ["x1", "x2", "x5", "x4", "x3", "x6"]
        )
        assert xg.get("x2").tops == lpx.StrVector(["x3"])
        assert xg.get("x2").bottoms == lpx.StrVector(["x1"])
        assert xg.get("x4").tops == lpx.StrVector(["x3", "x6"])
        assert xg.get("x4").bottoms == lpx.StrVector(["x1", "x5"])

    def test_get(self):
        xg = lpx.XGraph("g")
        X1 = lpx.XLayer(
            name="x1", xtype=lpx.StrVector(["X1"]), bottoms=lpx.StrVector([])
        )

        xg.add(X1)

        X1_xg = xg.get("x1")
        assert X1_xg.name == "x1"
        assert X1_xg.xtype[0] == "X1"
        assert X1_xg.bottoms == lpx.StrVector([])

        # If we adjust X1, this doesn't get represented in X1_xg
        X1.xtype[0] = "X11"
        assert X1_xg.xtype[0] == "X1"

        X1_xg.xtype[0] = "X11"
        assert X1_xg.xtype[0] == "X11"

        X1_xg.xtype = lpx.StrVector(["X111"])
        assert X1_xg.xtype[0] == "X111"

    def test_remove(self):
        xg = lpx.XGraph("g")
        X1 = lpx.XLayer(
            name="x1", xtype=lpx.StrVector(["Input"]), bottoms=lpx.StrVector([])
        )

        xg.add(X1)
        assert "x1" in xg
        assert len(xg) == 1

        xg.remove("x1")
        assert len(xg) == 0

        X2 = lpx.XLayer(
            name="x2", xtype=lpx.StrVector(["X2"]), bottoms=lpx.StrVector(["x1"])
        )
        X3 = lpx.XLayer(
            name="x3", xtype=lpx.StrVector(["X3"]), bottoms=lpx.StrVector(["x2"])
        )
        X4 = lpx.XLayer(
            name="x4",
            xtype=lpx.StrVector(["X4"]),
            bottoms=lpx.StrVector(["x1"]),
            tops=lpx.StrVector(["x3"]),
        )
        X5 = lpx.XLayer(
            name="x5",
            xtype=lpx.StrVector(["Input"]),
            bottoms=lpx.StrVector([]),
            tops=lpx.StrVector(["x4"]),
        )
        X6 = lpx.XLayer(
            name="x6",
            xtype=lpx.StrVector(["X6"]),
            bottoms=lpx.StrVector(["x4"]),
            tops=lpx.StrVector([]),
        )

        xg.add(X1)
        xg.add(X2)
        xg.add(X3)
        xg.add(X4)
        xg.add(X5)
        xg.add(X6)

        assert len(xg) == 6

        xg.remove("x2")
        assert len(xg) == 5
        assert xg.get_input_names() == lpx.StrVector(["x1", "x5"])
        assert xg.get_output_names() == lpx.StrVector(["x3", "x6"])
        assert xg.get_layer_names() == lpx.StrVector(["x1", "x5", "x4", "x3", "x6"])

        xg.remove("x1")
        assert len(xg) == 4
        assert xg.get_input_names() == lpx.StrVector(["x5"])
        assert xg.get_output_names() == lpx.StrVector(["x3", "x6"])
        assert xg.get_layer_names() == lpx.StrVector(["x5", "x4", "x3", "x6"])

        xg.remove("x6")
        assert len(xg) == 3
        assert xg.get_input_names() == lpx.StrVector(["x5"])
        assert xg.get_output_names() == lpx.StrVector(["x3"])
        assert xg.get_layer_names() == lpx.StrVector(["x5", "x4", "x3"])

        xg.remove("x4")
        assert len(xg) == 2
        assert xg.get_input_names() == lpx.StrVector(["x5", "x3"])
        assert xg.get_output_names() == lpx.StrVector(["x3", "x5"])
        assert xg.get_layer_names() == lpx.StrVector(["x3", "x5"])

        xg.remove("x3")
        assert len(xg) == 1
        assert xg.get_input_names() == lpx.StrVector(["x5"])
        assert xg.get_output_names() == lpx.StrVector(["x5"])
        assert xg.get_layer_names() == lpx.StrVector(["x5"])

        xg.remove("x5")
        assert len(xg) == 0
        assert xg.get_input_names() == lpx.StrVector([])
        assert xg.get_output_names() == lpx.StrVector([])
        assert xg.get_layer_names() == lpx.StrVector([])

    def test_update(self):
        xg = lpx.XGraph("g")
        X1 = lpx.XLayer(
            name="x1", xtype=lpx.StrVector(["X1"]), bottoms=lpx.StrVector([])
        )

        xg.add(X1)
        assert xg.get("x1").xtype == lpx.StrVector(["X1"])

        X1.xtype[0] = "X11"
        assert xg.get("x1").xtype == lpx.StrVector(["X1"])

        xg.update(X1.name)
        assert xg.get("x1").xtype == lpx.StrVector(["X1"])

        X2 = lpx.XLayer(
            name="x2", xtype=lpx.StrVector(["X2"]), bottoms=lpx.StrVector(["x1"])
        )
        X3 = lpx.XLayer(
            name="x3", xtype=lpx.StrVector(["X3"]), bottoms=lpx.StrVector(["x2"])
        )

        xg.add(X2)
        xg.add(X3)
        X2 = xg.get("x2")
        X3 = xg.get("x3")

        assert xg.get_layer_names() == lpx.StrVector(["x1", "x2", "x3"])
        X3.bottoms = lpx.StrVector(["x2"])
        xg.update(X3.name)
        assert xg.get_layer_names() == lpx.StrVector(["x1", "x2", "x3"])
        assert xg.get("x2").tops == lpx.StrVector(["x3"])
        assert xg.get("x2").bottoms == lpx.StrVector(["x1"])
        assert xg.get("x3").bottoms == lpx.StrVector(["x2"])
        assert xg.get("x1").tops == lpx.StrVector(["x2"])

        xg.remove(X2.name)
        X2.bottoms = lpx.StrVector(["x3"])
        X2.tops = lpx.StrVector(["x1"])
        xg.add(X2)
        assert xg.get_layer_names() == lpx.StrVector(["x3", "x2", "x1"])
        assert xg.get_input_names() == lpx.StrVector(["x3"])
        assert xg.get_output_names() == lpx.StrVector(["x1"])
        assert xg.get("x2").tops == lpx.StrVector(["x1"])
        assert xg.get("x2").bottoms == lpx.StrVector(["x3"])
        assert xg.get("x3").bottoms == lpx.StrVector([])
        assert xg.get("x3").tops == lpx.StrVector(["x2"])
        assert xg.get("x1").tops == lpx.StrVector([])
        assert xg.get("x1").bottoms == lpx.StrVector(["x2"])
