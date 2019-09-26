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
Module for testing the Vector data structure wrappers around
C++ libpyxir data structures


"""

import unittest
import numpy as np
import libpyxir as lpx

from pyxir.shared.vector import StrVector, IntVector, FloatVector,\
    IntVector2D


class TestVector(unittest.TestCase):

    def test_int_vector(self):

        iv = lpx.IntVector([1, 2, 3])
        ivx = IntVector(iv)
        assert ivx == iv
        assert ivx == [1, 2, 3]

        # List comprehension
        assert [i for i in ivx] == [1, 2, 3]

        # Append
        ivx.append(4)
        assert len(iv) == 4
        assert len(ivx) == 4

        # Contains
        assert 2 in iv
        assert 2 in ivx
        assert -1 not in iv
        assert -1 not in ivx
        with self.assertRaises(TypeError):
            assert 'a' not in ivx

        # Delete
        del ivx[3]
        assert ivx == [1, 2, 3]
        assert iv == lpx.IntVector([1, 2, 3])

        # Equal
        assert ivx == [1, 2, 3]
        assert ivx == lpx.IntVector([1, 2, 3])
        assert iv == lpx.IntVector([1, 2, 3])

        # Add / Extend
        ivx.extend([4, 5])
        assert len(iv) == 5
        assert len(ivx) == 5

        ivx += [6, 7]
        assert len(ivx) == 7
        del ivx[6]
        del ivx[5]

        # Get item
        assert ivx[3] == 4
        assert ivx[-1] == 5
        assert len(ivx) == 5
        assert ivx[:] == [1, 2, 3, 4, 5]
        with self.assertRaises(IndexError):
            ivx[6]

        # Index
        assert ivx.index(2) == 1
        assert ivx.index(5) == 4
        with self.assertRaises(ValueError):
            ivx.index('a')

        # Insert
        ivx.insert(0, -1)
        assert len(ivx) == 6
        assert len(iv) == 6
        assert ivx[0] == -1
        del ivx[0]

        # Iter
        c = [1, 2, 3, 4, 5]
        for i, e in enumerate(ivx):
            assert e == c[i]
        for i, e in enumerate(iv):
            assert e == c[i]

        # Length
        assert len(ivx) == len(iv)
        assert len(ivx) == 5

        # Not equal
        assert iv != lpx.IntVector([1, 2, 3])
        assert ivx != [1, 2, 3, 4]

        # Repr
        assert repr(iv) == "IntVector[1, 2, 3, 4, 5]"
        assert repr(ivx) == "IntVector[1, 2, 3, 4, 5]"

        # Str
        assert str(iv) == "IntVector[1, 2, 3, 4, 5]"
        assert str(ivx) == "IntVector[1, 2, 3, 4, 5]"

        # Set
        ivx[0] = -1
        assert ivx == [-1, 2, 3, 4, 5]
        assert iv == lpx.IntVector([-1, 2, 3, 4, 5])
        with self.assertRaises(IndexError):
            ivx[6] = -1
        ivx[:] = [-1, -2, -3, -4, -5]
        assert len(ivx) == 5
        assert ivx == [-1, -2, -3, -4, -5]

    def test_int_vector_2D(self):

        iv = lpx.IntVector2D([lpx.IntVector([])])
        ivx = IntVector2D(iv)
        assert ivx == iv
        assert ivx == [[]]

        # Append
        assert len(iv) == 1
        assert len(ivx) == 1
        ivx.append([4, 5, 6])
        assert len(iv) == 2
        assert len(ivx) == 2
        assert ivx == [[], [4, 5, 6]]
        ivx[0].append(1)
        ivx[0].append(2)
        ivx[0].append(3)
        assert len(iv) == 2
        assert len(ivx) == 2
        assert ivx == [[1, 2, 3], [4, 5, 6]]

        # Contains
        assert lpx.IntVector([1, 2, 3]) in iv
        assert lpx.IntVector([4, 5, 6]) in iv
        assert [1, 2, 3] in ivx
        assert [4, 5, 6] in ivx
        assert lpx.IntVector([1, 2]) not in iv
        assert [1, 2] not in ivx
        with self.assertRaises(TypeError):
            assert 1 not in ivx
        with self.assertRaises(TypeError):
            assert 'a' not in ivx

        # Delete
        with self.assertRaises(IndexError):
            del ivx[2]
        del ivx[1]
        assert ivx == [[1, 2, 3]]
        assert iv == lpx.IntVector2D([lpx.IntVector([1, 2, 3])])

        # Equal
        assert ivx == [[1, 2, 3]]
        assert ivx == lpx.IntVector2D([lpx.IntVector([1, 2, 3])])
        assert iv == lpx.IntVector2D([lpx.IntVector([1, 2, 3])])

        # Extend
        ivx.extend([[4, 5]])
        assert len(iv) == 2
        assert len(ivx) == 2

        ivx += [[6, 7]]
        assert len(ivx) == 3
        del ivx[2]

        # Get item
        assert ivx[0] == [1, 2, 3]
        assert ivx[-1] == [4, 5]
        with self.assertRaises(IndexError):
            ivx[6]
        assert ivx[0:1] == [[1, 2, 3]]
        assert ivx[:] == [[1, 2, 3], [4, 5]]

        # Insert
        ivx.insert(0, [0, 0])
        assert len(ivx) == 3
        assert len(iv) == 3
        assert ivx[0] == [0, 0]
        del ivx[0]
        assert len(ivx) == 2

        # Iter
        c = [[1, 2, 3], [4, 5]]
        for i, e in enumerate(ivx):
            assert isinstance(e, IntVector)
            assert e == c[i]
        for i, e in enumerate(iv):
            assert isinstance(e, lpx.IntVector)
            assert e == lpx.IntVector(c[i])

        # Length
        assert len(ivx) == len(iv)
        assert len(ivx) == 2

        # Not equal
        assert iv != lpx.IntVector2D([lpx.IntVector([1, 2, 3])])
        assert ivx != [[1, 2, 3], [4]]

        # Repr
        assert repr(ivx) == "IntVector2D[IntVector[1, 2, 3], IntVector[4, 5]]"

        # Str
        assert str(ivx) == "IntVector2D[IntVector[1, 2, 3], IntVector[4, 5]]"

        # Set
        ivx[0] = [-1, 2, 3]
        assert ivx == [[-1, 2, 3], [4, 5]]
        assert iv == lpx.IntVector2D([lpx.IntVector([-1, 2, 3]),
                                      lpx.IntVector([4, 5])])
        with self.assertRaises(IndexError):
            ivx[6] = [5, 6]

    def test_float_vector(self):

        iv = lpx.FloatVector([1, 1.5, 3])
        ivx = FloatVector(iv)
        assert ivx == iv
        assert ivx == [1, 1.5, 3]

        # Append
        ivx.append(4)
        assert len(iv) == 4
        assert len(ivx) == 4

        # Contains
        assert 1.5 in iv
        assert 1.5 in ivx
        assert 1.51 not in iv
        assert 1.51 not in ivx
        with self.assertRaises(TypeError):
            assert 'a' not in ivx

        # Delete
        del ivx[3]
        assert ivx == [1,  1.5, 3]
        assert iv == lpx.FloatVector([1, 1.5, 3])

        # Equal
        assert ivx == [1, 1.5, 3]
        assert ivx == lpx.FloatVector([1, 1.5, 3])
        assert iv == lpx.FloatVector([1, 1.5, 3])

        # Extend
        ivx.extend([4, 5])
        assert len(iv) == 5
        assert len(ivx) == 5

        ivx += [6., 6.5]
        assert len(iv) == 7
        assert len(ivx) == 7
        del ivx[5]
        del ivx[5]

        # Get item
        assert ivx[3] == 4
        assert ivx[-1] == 5
        with self.assertRaises(IndexError):
            ivx[6]

        # Iter
        c = [1, 1.5, 3, 4, 5]
        for i, e in enumerate(ivx):
            assert e == c[i]
        for i, e in enumerate(iv):
            assert e == c[i]

        # Length
        assert len(ivx) == len(iv)
        assert len(ivx) == 5

        # Not equal
        assert iv != lpx.FloatVector([1, 1.5, 3])
        assert ivx != [1, 1.5, 3, 4]

        # Repr
        assert repr(iv) == "FloatVector[1, 1.5, 3, 4, 5]"
        assert repr(ivx) == "FloatVector[1, 1.5, 3, 4, 5]"

        # Str
        assert str(iv) == "FloatVector[1, 1.5, 3, 4, 5]"
        assert str(ivx) == "FloatVector[1, 1.5, 3, 4, 5]"

        # Set
        ivx[0] = -1
        assert ivx == [-1, 1.5, 3, 4, 5]
        assert iv == lpx.FloatVector([-1, 1.5, 3, 4, 5])
        with self.assertRaises(IndexError):
            ivx[6] = -1

        # to list
        assert ivx.to_list() == [-1, 1.5, 3, 4, 5]

    def test_str_vector(self):

        iv = lpx.StrVector(['a', 'b', 'c'])
        ivx = StrVector(iv)
        assert ivx == iv
        assert ivx == ['a', 'b', 'c']

        # Append
        ivx.append('d')
        assert len(iv) == 4
        assert len(ivx) == 4

        # Contains
        assert 'b' in iv
        assert 'b' in ivx
        assert 'e' not in iv
        assert 'e' not in ivx

        # Delete
        del ivx[3]
        assert ivx == ['a', 'b', 'c']
        assert iv == lpx.StrVector(['a', 'b', 'c'])

        # Equal
        assert ivx == ['a', 'b', 'c']
        assert ivx == lpx.StrVector(['a', 'b', 'c'])
        assert iv == lpx.StrVector(['a', 'b', 'c'])

        # Extend
        ivx.extend(['d', 'e'])
        assert len(iv) == 5
        assert len(ivx) == 5

        # Get item
        assert ivx[3] == 'd'
        assert ivx[-1] == 'e'
        with self.assertRaises(IndexError):
            ivx[6]

        # Iter
        c = ['a', 'b', 'c', 'd', 'e']
        for i, e in enumerate(ivx):
            assert e == c[i]
        for i, e in enumerate(iv):
            assert e == c[i]

        # Length
        assert len(ivx) == len(iv)
        assert len(ivx) == 5

        # Not equal
        assert iv != lpx.StrVector(['a', 'b', 'c'])
        assert ivx != ['a', 'b', 'c', 'd']

        # Repr
        assert repr(iv) == "StrVector[a, b, c, d, e]"
        assert repr(ivx) == "StrVector[a, b, c, d, e]"

        # Str
        assert str(iv) == "StrVector[a, b, c, d, e]"
        assert str(ivx) == "StrVector[a, b, c, d, e]"

        # Set
        ivx[0] = 'z'
        assert ivx == ['z', 'b', 'c', 'd', 'e']
        assert iv == lpx.StrVector(['z', 'b', 'c', 'd', 'e'])
        with self.assertRaises(IndexError):
            ivx[6] = 'z'

    # def test_xbuffer_vector(self):

    #     iv = lpx.XBufferVector([np.array(1, 2, 3)])
    #     ivx = FloatVector(iv)
    #     assert ivx == iv
    #     assert ivx == [1, 1.5, 3]

    #     # Append
    #     ivx.append(4)
    #     assert len(iv) == 4
    #     assert len(ivx) == 4

    #     # Contains
    #     assert 1.5 in iv
    #     assert 1.5 in ivx
    #     assert 1.51 not in iv
    #     assert 1.51 not in ivx
    #     with self.assertRaises(TypeError):
    #         assert 'a' not in ivx

    #     # Delete
    #     del ivx[3]
    #     assert ivx == [1,  1.5, 3]
    #     assert iv == lpx.FloatVector([1, 1.5, 3])

    #     # Equal
    #     assert ivx == [1, 1.5, 3]
    #     assert ivx == lpx.FloatVector([1, 1.5, 3])
    #     assert iv == lpx.FloatVector([1, 1.5, 3])

    #     # Extend
    #     ivx.extend([4, 5])
    #     assert len(iv) == 5
    #     assert len(ivx) == 5

    #     # Get item
    #     assert ivx[3] == 4
    #     assert ivx[-1] == 5
    #     with self.assertRaises(IndexError):
    #         ivx[6]

    #     # Iter
    #     c = [1, 1.5, 3, 4, 5]
    #     for i, e in enumerate(ivx):
    #         assert e == c[i]
    #     for i, e in enumerate(iv):
    #         assert e == c[i]

    #     # Length
    #     assert len(ivx) == len(iv)
    #     assert len(ivx) == 5

    #     # Not equal
    #     assert iv != lpx.FloatVector([1, 1.5, 3])
    #     assert ivx != [1, 1.5, 3, 4]

    #     # Repr
    #     assert repr(iv) == "FloatVector[1, 1.5, 3, 4, 5]"
    #     assert repr(ivx) == "FloatVector[1, 1.5, 3, 4, 5]"

    #     # Str
    #     assert str(iv) == "FloatVector[1, 1.5, 3, 4, 5]"
    #     assert str(ivx) == "FloatVector[1, 1.5, 3, 4, 5]"

    #     # Set
    #     ivx[0] = -1
    #     assert ivx == [-1, 1.5, 3, 4, 5]
    #     assert iv == lpx.FloatVector([-1, 1.5, 3, 4, 5])
    #     with self.assertRaises(IndexError):
    #         ivx[6] = -1