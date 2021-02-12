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

try:
    from pyxir.runtime.tensorflow.x_2_tf_registry import \
        X_2_TF, rt_register_xlayer_2_tf
    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestX2TfRegistry(unittest.TestCase):

    @unittest.skipIf(skip_tf, "Skipping Tensorflow related test because Tensorflow is not available")
    def test_x_2_tf(self):
        # L0: Inputs & Other
        assert 'Constant' in X_2_TF
        assert 'Input' in X_2_TF
        assert 'Output' in X_2_TF
        assert 'StrInput' in X_2_TF
        assert 'Tuple' in X_2_TF
        assert 'Variable' in X_2_TF

        # L1: Basic NN
        assert 'BiasAdd' in X_2_TF
        assert 'Concat' in X_2_TF
        assert 'Dense' in X_2_TF
        assert 'Eltwise' in X_2_TF
        assert 'Pad' in X_2_TF
        assert 'pReLU' in X_2_TF
        assert 'ReLU' in X_2_TF
        assert 'ReLU6' in X_2_TF
        assert 'Scale' in X_2_TF
        assert 'Softmax' in X_2_TF
        assert 'Tanh' in X_2_TF

        # L2: Convolutions
        assert 'BatchNorm' in X_2_TF
        assert 'Convolution' in X_2_TF
        assert 'Conv2DTranspose' in X_2_TF
        assert 'Flatten' in X_2_TF
        assert 'Pooling' in X_2_TF
        assert 'PoolingNoDivision' in X_2_TF

        # L3: Math and Transformations
        assert 'Reshape' in X_2_TF
        assert 'Squeeze' in X_2_TF
        assert 'Transpose' in X_2_TF

        # L4: Broadcast and Reductions
        assert 'Mean' in X_2_TF

        # L5: Vision
        assert 'Cvx' in X_2_TF

        # L11: Quantization
        assert 'Quantize' in X_2_TF
        assert 'UnQuantize' in X_2_TF
        assert 'QuantizeBias' in X_2_TF
        assert 'QuantizeScaleBias' in X_2_TF
        assert 'QuantizeInter' in X_2_TF

    @unittest.skipIf(skip_tf, "Skipping Tensorflow related test because Tensorflow is not available")
    def test_register_xlayer_2_tf(self):

        with self.assertRaises(ValueError):

            @rt_register_xlayer_2_tf('Dense')
            class Test(object):
                pass
