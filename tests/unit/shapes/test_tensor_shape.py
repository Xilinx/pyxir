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
Module for testing the TensorShape data structure


"""

import unittest
import libpyxir as lpx

from pyxir.shapes import TensorShape
from pyxir.shared.vector import IntVector


class TestTensorShape(unittest.TestCase):

    def test_get_set_item(self):

        ts = TensorShape(IntVector(lpx.IntVector([-1, 2])))

        assert len(ts) == 2
        assert isinstance(ts[0], int)
        assert ts[0] == -1
        assert ts[1] == 2

        with self.assertRaises(IndexError):
            ts[3]

        ts[1] = 3
        assert ts[1] == 3

    def test_get_set_item_slice(self):

        ts = TensorShape(IntVector(lpx.IntVector([-1, 2, 3, 4])))

        ts_slice = ts[1:3]

        assert len(ts_slice) == 2
        assert ts_slice == [2, 3]

        ts[1:3] = [3, 2]
        assert ts == [-1, 3, 2, 4]

    def test_get_size(self):

        ts = TensorShape(IntVector(lpx.IntVector([-1, 2, 3, 4])))

        assert ts.get_size() == [24]

    def test_set_value(self):

        ts = TensorShape(IntVector(lpx.IntVector([1, 2, 3, 4])))

        ts.set_value(0, -1)

        assert len(ts) == 4
        assert ts[0] == -1
        assert ts[1] == 2

    def test_to_list(self):

        ts = TensorShape(IntVector(lpx.IntVector([1, 2, 3, 4])))

        assert ts == [1, 2, 3, 4]
        lst = ts.tolist()
        lst2 = list(ts)

        ts[0] = -1
        assert ts == [-1, 2, 3, 4]
        assert lst == [1, 2, 3, 4]
        assert lst2 == [1, 2, 3, 4]
