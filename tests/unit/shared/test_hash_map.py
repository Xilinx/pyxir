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
Module for testing the HashMap data structure wrappers around
C++ libpyxir data structures


"""

import unittest
import numpy as np
import libpyxir as lpx

from pyxir.shared.hash_map import MapStrVectorStr
from pyxir.shared.vector import StrVector


class TestHashMap(unittest.TestCase):

    def test_map_str_vector_str_from_dict(self):
        m = MapStrVectorStr.from_dict({'a': ['a', 'b'], 'b': ['b']})
        assert len(m) == 2
        assert m['a'] == ['a', 'b']
        assert m['b'] == ['b']

    def test_map_str_vector_str(self):

        # Constructor

        _m = lpx.MapStrVectorStr()
        _m['a'] = lpx.StrVector(['a', 'b'])
        m = MapStrVectorStr(_m)
        assert len(m) == 1

        # Contains
        assert 'a' in m
        assert 'b' not in m
        with self.assertRaises(TypeError):
            assert 1 not in m

        _m['b'] = lpx.StrVector(['b', 'b'])

        # Delete
        assert len(m) == 2
        del m['b']
        assert len(m) == 1
        assert m['a'] == ['a', 'b']
        _m['c'] = lpx.StrVector(['c', 'c'])

        # Equal
        assert m == {'a': ['a', 'b'], 'c': ['c', 'c']}
        assert m == {'a': lpx.StrVector(['a', 'b']),
                     'c': lpx.StrVector(['c', 'c'])}

        # Get item
        assert m['a'] == ['a', 'b']
        assert m['a'] == lpx.StrVector(['a', 'b'])
        assert m.get('b') is None
        with self.assertRaises(KeyError):
            m['b']

        # Get lpx map
        assert m.get_lpx_map() == _m

        # Items
        assert ('a', ['a', 'b']) in m.items()
        assert ('c', ('c', 'c')) in m.items()

        # Keys
        assert set(m.keys()) == set(['a', 'c'])

        # Length
        assert len(m) == 2
        assert len(_m) == 2

        # Not equal
        assert m != {'a': ['a', 'b']}

        # Set
        m['d'] = ['d']
        assert len(m) == 3
        m['a'] = lpx.StrVector(['a'])
        assert len(m) == 3
        assert m['a'] == ['a']
        m['c'][0] = 'cc'

        # to dict
        d = m.to_dict()
        assert len(d) == 3
        assert d['a'] == ['a']
        assert d['c'] == ['cc', 'c']
        d['c'][1] = 'cc'
        assert d['c'] == ['cc', 'cc']
        assert m['c'] == ['cc', 'c']
        assert d['d'] == ['d']

        # Update
        m.update({'a': ['update'], 'e': ['e']})
        assert len(m) == 4
        assert m['a'][0] == 'update'
        assert m['e'] == ['e']
