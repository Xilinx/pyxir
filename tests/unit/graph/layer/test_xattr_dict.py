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
Module for testing the XAttrDict wrapper data structure


"""

import unittest
import numpy as np
import libpyxir as lpx

from pyxir.graph.layer.xattr_dict import XAttrDict
from pyxir.shared.vector import IntVector, IntVector2D, FloatVector, StrVector


class TestXAttrDict(unittest.TestCase):

    def test_xattr(self):
        xa = lpx.XAttr('a', -1)
        assert xa.name == 'a'
        assert xa.type == 'INT'
        assert xa.i == -1

        xa = lpx.XAttr('b', lpx.IntVector([1, 2, 3]))
        assert xa.name == 'b'
        assert xa.type == 'INTS'
        assert xa.ints == lpx.IntVector([1, 2, 3])

        xa = lpx.XAttr('c', lpx.IntVector2D([lpx.IntVector([1, 2, 3]),
                                             lpx.IntVector([0, 0])]))
        assert xa.name == 'c'
        assert xa.type == 'INTS2D'
        assert xa.ints2d == lpx.IntVector2D([lpx.IntVector([1, 2, 3]),
                                             lpx.IntVector([0, 0])])

        xa = lpx.XAttr('d', -1.5)
        assert xa.name == 'd'
        assert xa.type == 'FLOAT'
        assert xa.f == -1.5

        xa = lpx.XAttr('e', lpx.FloatVector([1, 2, 3]))
        assert xa.name == 'e'
        assert xa.type == 'FLOATS'
        assert xa.floats == lpx.FloatVector([1, 2, 3])

        xa = lpx.XAttr('f', 'f')
        assert xa.name == 'f'
        assert xa.type == 'STRING'
        assert xa.s == 'f'

        xa = lpx.XAttr('e', lpx.StrVector(['a', 'b']))
        assert xa.name == 'e'
        assert xa.type == 'STRINGS'
        assert xa.strings == lpx.StrVector(['a', 'b'])

    def test_constructor(self):
        xam = lpx.XAttrMap()
        xad = XAttrDict(xam)
        assert len(xad) == 0
        assert len(xam) == 0

        xa = lpx.XAttr('a', -1)

        xam = lpx.XAttrMap()
        xam['a'] = xa
        xad = XAttrDict(xam)
        assert len(xad) == 1
        assert len(xam) == 1

    def test_clear(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', 'b')
        xad = XAttrDict(xam)
        assert len(xad) == 2
        assert len(xam) == 2

        xad.clear()
        assert len(xad) == 0
        assert len(xam) == 0

    def test_contains(self):
        xa = lpx.XAttr('a', -1)
        xam = lpx.XAttrMap()
        xam['a'] = xa
        xad = XAttrDict(xam)

        assert 'a' in xad
        assert 'b' not in xad

    def test_copy(self):
        xa = lpx.XAttr('a', -1)
        xam = lpx.XAttrMap()
        xam['a'] = xa
        xad = XAttrDict(xam)

        xad['ints'] = [1, 2]
        xad['strings'] = ['a', 'b']
        xad['ints2d'] = [[1, 2], [3, 4]]
        xad['map_str_vstr'] = {'a': ['a1', 'a2'], 'b': ['b1', 'b2']}

        xad2 = xad.copy()

        assert 'a' in xad
        assert 'a' in xad2
        assert xad2['a'] == -1
        assert xad2['map_str_vstr'] == {'a': ['a1', 'a2'], 'b': ['b1', 'b2']}

        del xad['a']
        xad['map_str_vstr']['a'][0] = 'aa'
        xad['ints2d'][0] = [2, 1]

        assert 'a' not in xad
        assert 'a' in xad2

        assert xad['ints2d'] == [[2, 1], [3, 4]]
        assert xad2['ints2d'] == [[1, 2], [3, 4]]

        assert xad['map_str_vstr'] == {'a': ['aa', 'a2'], 'b': ['b1', 'b2']}
        assert xad2['map_str_vstr'] == {'a': ['a1', 'a2'], 'b': ['b1', 'b2']}

    def test_del(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', 1)
        xad = XAttrDict(xam)

        assert len(xad) == 2
        del xad['b']
        assert len(xad) == 1
        assert 'a' in xad
        assert xad['a'] == -1

    def test_eq(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', 1)
        xad = XAttrDict(xam)

        xad2 = xad.copy()

        assert xad == xad2

        xam3 = lpx.XAttrMap()
        xam3['a'] = lpx.XAttr('a', -1)
        xam3['b'] = lpx.XAttr('b', 1)
        xad3 = XAttrDict(xam3)

        assert xad == xad3
        assert xad2 == xad3

    def test_get(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xb = lpx.XAttr('b', lpx.IntVector([1, 2]))
        xam['b'] = xb
        xad = XAttrDict(xam)

        assert xad.get('a') == -1
        # assert xad.get('b') == [1, 2]
        assert xad.get('c') is None

    def test_getitem(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.IntVector([1, 2]))
        xam['c'] = lpx.XAttr('c', lpx.IntVector2D([lpx.IntVector([1, 2]),
                                                   lpx.IntVector([0, 0])]))
        xam['d'] = lpx.XAttr('d', -1.5)
        xam['e'] = lpx.XAttr('e', lpx.FloatVector([1.5, 2]))
        xam['f'] = lpx.XAttr('f', 'f')
        xam['g'] = lpx.XAttr('g', lpx.StrVector(['gg', 'gg']))
        xam['h'] = lpx.XAttr('h', True)
        xam['i'] = lpx.XAttr('i', lpx.MapStrVectorStr())
        xad = XAttrDict(xam)
        xad['i'] = {'i': ['i1', 'i2']}
        xad['j'] = {'j': 'j1'}

        assert xad['a'] == -1
        assert xad['b'] == [1, 2]
        assert xad['c'] == [[1, 2], [0, 0]]
        assert xad['d'] == -1.5
        assert xad['e'] == [1.5, 2]
        assert xad['f'] == 'f'
        assert xad['g'] == ['gg', 'gg']
        assert xad['h'] is True
        assert xad['i'] == {'i': ['i1', 'i2']}
        assert xad['i'] != {'a': ['a']}
        assert xad['j'] == {'j': 'j1'}

        with self.assertRaises(KeyError):
            xad['z']

    def test_get_copy(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.IntVector([1, 2]))
        xam['c'] = lpx.XAttr('c', lpx.IntVector2D([lpx.IntVector([1, 2]),
                                                   lpx.IntVector([0, 0])]))
        xam['d'] = lpx.XAttr('d', -1.5)
        xam['e'] = lpx.XAttr('e', lpx.FloatVector([1.5, 2]))
        xam['f'] = lpx.XAttr('f', 'f')
        xam['g'] = lpx.XAttr('g', lpx.StrVector(['gg', 'gg']))
        xam['h'] = lpx.XAttr('h', False)
        xad = XAttrDict(xam)

        a_c = xad._get_copy('a')
        assert a_c == -1
        xad['a'] = 1
        assert a_c == -1
        assert xad['a'] == 1

        ints_c = xad._get_copy('b')
        assert ints_c == [1, 2]
        xad['b'][0] = -1
        assert xad['b'] == [-1, 2]
        assert ints_c == [1, 2]

        ints2d_c = xad._get_copy('c')
        assert ints2d_c == [[1, 2], [0, 0]]
        xad['c'][0] = [-1, 2]
        assert xad['c'] == [[-1, 2], [0, 0]]
        assert ints2d_c == [[1, 2], [0, 0]]

        assert xad._get_copy('d') == -1.5

        floats_c = xad._get_copy('e')
        assert floats_c == [1.5, 2]
        xad['e'] = [-1.5, 2.]
        assert xad['e'] == [-1.5, 2]
        assert floats_c == [1.5, 2]

        s_c = xad._get_copy('f')
        assert s_c == 'f'
        xad['f'] = 'ff'
        assert xad['f'] == 'ff'
        assert s_c == 'f'

        strings_c = xad._get_copy('g')
        assert strings_c == ['gg', 'gg']
        xad['g'] = ['gg']
        assert xad['g'] == ['gg']
        assert strings_c == ['gg', 'gg']

        b_c = xad._get_copy('h')
        assert b_c is False
        xad['h'] = True
        assert xad['h'] is True
        assert b_c is False

        with self.assertRaises(KeyError):
            xad._get_copy('z')

    def test_get_xattr_map(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xad = XAttrDict(xam)

        assert xad._get_xattr_map() == xam

    def test_items(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.FloatVector([1.5, 2]))
        xad = XAttrDict(xam)

        assert ('a', -1) in xad.items()
        assert ('b', [1.5, 2.]) in xad.items()
        assert ('c', 0) not in xad.items()

    def test_keys(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.FloatVector([1.5, 2]))
        xad = XAttrDict(xam)

        assert set(xad.keys()) == set(['a', 'b'])

    def test_len(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.FloatVector([1.5, 2]))
        xad = XAttrDict(xam)

        assert len(xad) == 2
        assert len(xam) == 2

        xam['c'] = lpx.XAttr('c', 'c')
        assert len(xad) == 3
        del xad['c']
        assert len(xad) == 2
        xad.clear()
        assert len(xad) == 0

    def test_pop(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.FloatVector([1.5, 2]))
        xad = XAttrDict(xam)

        assert len(xad) == 2
        a = xad.pop('a')
        assert len(xad) == 1
        assert a == -1
        b = xad.pop('b')
        assert len(xad) == 0
        assert b == [1.5, 2]

    def test_setitem(self):
        xam = lpx.XAttrMap()
        xad = XAttrDict(xam)
        assert len(xad) == 0

        xad['i'] = 1
        assert xad['i'] == 1
        xad['i'] = -1
        assert xad['i'] == -1

        xad['ints'] = [1, 2]
        assert xad['ints'] == [1, 2]
        xad['ints'][0] = -1
        assert xad['ints'] == [-1, 2]
        xad['ints'] = [3, 3, 3]
        assert xad['ints'] == [3, 3, 3]

        xad['ints2d'] = [[1, 2], [0, 0]]
        assert xad['ints2d'] == [[1, 2], [0, 0]]
        xad['ints2d'][0][0] = -1
        assert xad['ints2d'] == [[-1, 2], [0, 0]]
        xad['ints2d'][0] = [3, 3]
        assert xad['ints2d'] == [[3, 3], [0, 0]]
        xad['ints2d'] = [[3, 3, 3]]
        assert xad['ints2d'] == [[3, 3, 3]]

        xad['f'] = 1.
        assert xad['f'] == 1.
        xad['f'] = -1.
        assert xad['f'] == -1.

        xad['floats'] = [1., 2.]
        assert xad['floats'] == [1., 2.]
        xad['floats'][0] = -1.
        assert xad['floats'] == [-1., 2.]
        xad['floats'] = [3., 3., 3.]
        assert xad['floats'] == [3., 3., 3.]

        xad['s'] = 's'
        assert xad['s'] == 's'
        xad['s'] = 't'
        assert xad['s'] == 't'

        xad['strings'] = ['a', 'b']
        assert xad['strings'] == ['a', 'b']
        xad['strings'][0] = 'z'
        assert xad['strings'] == ['z', 'b']
        xad['strings'] = ['s', 't']
        assert xad['strings'] == ['s', 't']

        xad['m'] = {'a': ['a', 'b'], 'b': ['b']}
        assert xad['m'] == {'a': ['a', 'b'], 'b': ['b']}
        xad['m']['a'] = ['set']
        assert xad['m'] == {'a': ['set'], 'b': ['b']}
        xad['m']['a'].insert(1, 'b')
        assert xad['m'] == {'a': ['set', 'b'], 'b': ['b']}
        del xad['m']['a'][1]
        assert xad['m'] == {'a': ['set'], 'b': ['b']}

    def test_update(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.FloatVector([1.5, 2]))
        xad = XAttrDict(xam)

        xad2 = XAttrDict(lpx.XAttrMap())
        xad2['b'] = [-1.5, -2.]
        xad2['c'] = ['a', 'b']

        xad.update(xad2)

        assert len(xad) == 3
        assert xad['a'] == -1
        assert xad['b'] == [-1.5, -2.]
        assert xad['c'] == ['a', 'b']

        xad.update({'d': [[1, 2], [3, 4]]})
        assert len(xad) == 4
        assert xad['d'] == [[1, 2], [3, 4]]

        # Change type of 'b' attribute and add another
        xad.update({'b': ['a', 'b', 'c'], 'e': [1, 2]})
        assert len(xad) == 5
        assert xad['b'] == ['a', 'b', 'c']
        assert xad['e'] == [1, 2]

    def test_values(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.FloatVector([1.5, 2]))
        xad = XAttrDict(xam)

        assert -1 in xad.values()
        assert [1.5, 2] in xad.values()

    def test_to_dict(self):
        xam = lpx.XAttrMap()
        xam['a'] = lpx.XAttr('a', -1)
        xam['b'] = lpx.XAttr('b', lpx.IntVector([1, 2]))
        xam['c'] = lpx.XAttr('c', lpx.IntVector2D([lpx.IntVector([1, 2]),
                                                   lpx.IntVector([0, 0])]))
        xam['d'] = lpx.XAttr('d', -1.5)
        xam['e'] = lpx.XAttr('e', lpx.FloatVector([1.5, 2]))
        xam['f'] = lpx.XAttr('f', 'f')
        xam['g'] = lpx.XAttr('g', lpx.StrVector(['gg', 'gg']))
        xam['h'] = lpx.XAttr('h', False)
        xad = XAttrDict(xam)
        xad['i'] = {'i': ['i1', 'i2']}
        xad['j'] = {'j': 'j1'}

        d = xad.to_dict()

        assert isinstance(d['a'], int)
        assert isinstance(d['b'], list)
        assert isinstance(d['c'], list)
        assert isinstance(d['d'], float)
        assert isinstance(d['e'], list)
        assert isinstance(d['f'], str)
        assert isinstance(d['g'], list)
        assert isinstance(d['h'], bool)
        assert isinstance(d['i'], dict)
        assert isinstance(d['i']['i'], list)
        assert isinstance(d['j'], dict)
        assert isinstance(d['j']['j'], str)
        assert d['j']['j'] == 'j1'
