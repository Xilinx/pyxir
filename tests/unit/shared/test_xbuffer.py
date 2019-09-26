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
Module for testing the XBuffer data structure wrappers around the C++ XBuffer


"""

import unittest
import numpy as np
import libpyxir as lpx

from pyxir.shared.xbuffer import XBuffer


class TestXBuffer(unittest.TestCase):

    def test_xbuffer_construction(self):
        a = np.array([1., 1.5], dtype=np.float32)
        xb = XBuffer(a)

        np.testing.assert_equal(xb.to_numpy(), a)

    def test_xbuffer_getattr(self):
        a = np.array([1., 1.5], dtype=np.float32)
        xb = XBuffer(a)

        assert xb.dtype == 'float32'
        assert xb.shape == (2,)

        a = np.zeros((1, 3, 224, 224), dtype=np.int8)
        xb = XBuffer(a)

        assert xb.dtype == 'int8'
        assert xb.shape == (1, 3, 224, 224)

    def test_xbuffer_operators(self):
        a = np.array([1., 1.5], dtype=np.float32)
        xb = XBuffer(a)

        xb1 = xb + [1, -1]
        assert isinstance(xb1, XBuffer)
        np.testing.assert_equal(xb.to_numpy(), a)
        np.testing.assert_equal(
            xb1.to_numpy(),
            np.array([2., .5], dtype=np.float32))

    def test_xbuffer_equals(self):
        a = np.array([1., 1.5], dtype=np.float32)
        xb = XBuffer(a)
        b = xb.to_numpy()

        xb *= np.array([2, -1])
        assert isinstance(xb, XBuffer)
        np.testing.assert_equal(
            xb.to_numpy(),
            np.array([2., -1.5], dtype=np.float32))
        np.testing.assert_equal(a, np.array([1., 1.5], dtype=np.float32))
        np.testing.assert_equal(b, np.array([1., 1.5], dtype=np.float32))

    def test_xbuffer_copy_from(self):
        a = np.array([1., 1.5], dtype=np.float32)
        xb = XBuffer(a)
        b = xb.to_numpy()

        xb.copy_from(xb * np.array([2, -1]))
        assert isinstance(xb, XBuffer)
        np.testing.assert_equal(
            xb.to_numpy(),
            np.array([2., -1.5], dtype=np.float32))
        np.testing.assert_equal(a, np.array([1., 1.5], dtype=np.float32))
        np.testing.assert_equal(b, np.array([2., -1.5], dtype=np.float32))
