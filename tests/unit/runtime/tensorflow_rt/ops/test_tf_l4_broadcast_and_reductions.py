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
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Tensorflow related test because Tensorflow is not available")

class TestTfL4BroadcastAndReduce(unittest.TestCase):

    def test_mean(self):

        X = xlayer.XLayer(
            type=['Mean'],
            name='mean1',
            shapes=[-1, 1, 1, 4],
            sizes=[4],
            bottoms=['in1'],
            tops=[],
            targets=[],
            attrs={
                'axes': [1, 2],
                'keepdims': True
            }
        )

        input_shapes = {'in1': TensorShape([1, 3, 3, 4])}

        layers = X_2_TF['Mean'](X, input_shapes, {})
        assert len(layers) == 1

        inpt = np.ones((1, 3, 3, 4))
        inputs = [inpt]

        out = layers[0].forward_exec(inputs)

        assert isinstance(out, np.ndarray)
        assert len(out) == 1
        assert out.shape == (1, 1, 1, 4)
