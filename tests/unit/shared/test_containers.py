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

""" Module for testing the StrContainer and BytesContainerdata structure """

import unittest
import libpyxir as lpx

from pyxir.shared.container import StrContainer, BytesContainer


class TestStrContainer(unittest.TestCase):

    def test_constructor(self):
        s = "test"
        sc = StrContainer(s)
        assert sc.get_str() == "test"

    def test_eq(self):
        s = "test"
        sc = StrContainer(s)
        assert sc == "test"

    def test_set_str(self):
        s = "test"
        sc = StrContainer(s)
        sc.set_str("2")
        assert sc == "2"
        assert sc.get_str() == "2"


class TestBytesContainer(unittest.TestCase):

    def test_constructor(self):
        b = b"test"
        bc = BytesContainer(b)
        assert isinstance(bc.get_bytes(), bytes)
        assert bc.get_bytes() == b"test"
        assert bc.get_bytes() != "test"

        b2 = "test".encode('latin1')
        bc2 = BytesContainer(b2)
        assert bc.get_bytes() == "test".encode('latin1')

    def test_eq(self):
        b = b"test"
        bc = BytesContainer(b)
        assert bc == b"test"

    def test_set_bytes(self):
        b = b"test"
        bc = BytesContainer(b)
        bc.set_bytes(b"2")
        assert bc == b"2"
        assert bc.get_bytes() == b"2"

    def test_set_bytes_latin1(self):
        b = b"test"
        bc = BytesContainer(b)
        bc.set_bytes("2".encode('latin1'))
        assert bc == "2".encode('latin1')
        assert bc.get_bytes() == "2".encode('latin1')
