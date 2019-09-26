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
Module for testing the TupleShape data structure


"""

import unittest
import libpyxir as lpx

from pyxir.shapes import TupleShape, TensorShape
from pyxir.shared.vector import IntVector, IntVector2D


class TestTupleShape(unittest.TestCase):

    def test_get_set_item(self):

        _shape = lpx.IntVector2D([lpx.IntVector([-1, 2]),
                                  lpx.IntVector([-1, 2, 4])])
        ts = TupleShape(IntVector2D(_shape))

        assert len(ts) == 2
        assert isinstance(ts[0], TensorShape)
        assert ts[0] == [-1, 2]
        assert ts[1] == [-1, 2, 4]

        with self.assertRaises(IndexError):
            ts[2]

        ts[0] = [-1, 3]
        assert ts == [[-1, 3], [-1, 2, 4]]

    def test_get_item_slice(self):

        ts = TupleShape([TensorShape([1]), TensorShape([2])])

        ts_slice = ts[:]

        assert len(ts_slice) == 2
        assert ts_slice == [[1], [2]]
        assert isinstance(ts_slice[0], TensorShape)
        assert isinstance(ts_slice[1], TensorShape)

        _shape = lpx.IntVector2D([lpx.IntVector([1]),
                                  lpx.IntVector([2])])
        ts = TupleShape(IntVector2D(_shape))

        ts_slice = ts[:]

        assert len(ts_slice) == 2
        assert ts_slice == [[1], [2]]
        assert isinstance(ts_slice[0], TensorShape)
        assert isinstance(ts_slice[1], TensorShape)

        ts[0:2] = [[-1, 2], [-1, 3]]

        assert ts == [[-1, 2], [-1, 3]]

    def test_get_size(self):

        ts = TupleShape([TensorShape([-1, 2]), TensorShape([-1, 2, 4])])

        assert ts.get_size() == [2, 8]

    def test_set_value(self):

        ts = TupleShape([TensorShape([1, 2]), TensorShape([2, 4])])

        ts.set_value(0, -1)

        assert len(ts) == 2
        assert ts[0] == TensorShape([-1, 2])
        assert ts[1] == TensorShape([-1, 4])

        _shape = lpx.IntVector2D([lpx.IntVector([1, 2]),
                                  lpx.IntVector([2, 4])])
        ts = TupleShape(IntVector2D(_shape))

        ts.set_value(0, -1)

        assert len(ts) == 2
        assert ts[0] == TensorShape([-1, 2])
        assert ts[1] == TensorShape([-1, 4])

    def test_to_list(self):

        ts = TupleShape([TensorShape([1, 2]), TensorShape([2, 4])])

        assert ts == [[1, 2], [2, 4]]
        assert ts.tolist() == [[1, 2], [2, 4]]

        _shape = lpx.IntVector2D([lpx.IntVector([1, 2]),
                                  lpx.IntVector([2, 4])])
        ts = TupleShape(IntVector2D(_shape))

        assert ts.tolist() == [[1, 2], [2, 4]]
