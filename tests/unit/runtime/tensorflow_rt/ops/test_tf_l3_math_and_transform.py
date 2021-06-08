#!/usr/bin/env python
#
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

"""Module for testing the pyxir TF executor"""

import unittest
import numpy as np

from pyxir.shapes import TensorShape
from pyxir.runtime import base
from pyxir.graph.layer import xlayer
from pyxir.graph.io import xlayer_io

try:
    from pyxir.runtime.tensorflow.x_2_tf_registry import X_2_TF
    from pyxir.runtime.tensorflow.ops.tf_l0_input_and_other import *
    from pyxir.runtime.tensorflow.ops.tf_l1_basic_nn import *
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Tensorflow related test because Tensorflow is not available")


class TestTfL3MathAndTransform(unittest.TestCase):

    def test_split_int(self):

        X = xlayer.XLayer(
            type=['Split'],
            name='split1',
            shapes=[[-1, 2, 4, 4], [-1, 2, 4, 4], [-1, 2, 4, 4]],
            sizes=[32, 32, 32],
            bottoms=['in1'],
            tops=[],
            targets=[],
            attrs={'axis': 1, 'indices': 3}
        )

        input_shapes = {'in1': TensorShape([1, 6, 4, 4])}

        layers = X_2_TF['Split'](X, input_shapes, {})
        assert len(layers) == 1

        inpt = np.ones((1, 6, 4, 4))
        inputs = [inpt]

        out = layers[0].forward_exec(inputs)

        assert isinstance(out, list)
        assert len(out) == 3

        assert out[0].shape == (1, 2, 4, 4)
        assert out[1].shape == (1, 2, 4, 4)
        assert out[2].shape == (1, 2, 4, 4)

    def test_split_tuple(self):

        X = xlayer.XLayer(
            type=['Split'],
            name='split1',
            shapes=[[-1, 1, 4, 4], [-1, 3, 4, 4], [-1, 1, 4, 4]],
            sizes=[32, 32, 32],
            bottoms=['in1'],
            tops=[],
            targets=[],
            attrs={'axis': 1, 'indices': [1, 4]}
        )

        input_shapes = {'in1': TensorShape([1, 5, 4, 4])}

        layers = X_2_TF['Split'](X, input_shapes, {})
        assert len(layers) == 1

        inpt = np.ones((1, 5, 4, 4))
        inputs = [inpt]

        out = layers[0].forward_exec(inputs)

        assert isinstance(out, list)
        assert len(out) == 3

        assert out[0].shape == (1, 1, 4, 4)
        assert out[1].shape == (1, 3, 4, 4)
        assert out[2].shape == (1, 1, 4, 4)

    def test_take(self):

        X = xlayer.XLayer(
            type=['Take'],
            name='take1',
            shapes=[-1, 1, 4],
            sizes=[4],
            bottoms=['in1', 'indices'],
            tops=[],
            targets=[],
            attrs={'axis': 1, 'mode': 'clip'}
        )

        input_shapes = {'in1': TensorShape([1, 3, 4]),
                        'indices': TensorShape([])}

        layers = X_2_TF['Take'](X, input_shapes, {})
        assert len(layers) == 1

        inpt = np.reshape(
            np.array([
                [[1, 1], [1, 1]],
                [[2, 2], [2, 2]],
                [[2, 2], [2, 2]]], dtype=np.float32),
            (1, 3, 4))
        indices = np.array(0, np.int32)
        inputs = [inpt, indices]

        out = layers[0].forward_exec(inputs)

        assert (out.shape) == (1, 4)
        np.testing.assert_array_equal(
            out, np.array([[1, 1, 1, 1]], dtype=np.float32))
