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

from pyxir.runtime import base
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.layer import xlayer

try:
    from pyxir.runtime.tensorflow.runtime_tf import *
    from pyxir.runtime.tensorflow import rt_layer_tf
    skip_tf = False
except ModuleNotFoundError:
    skip_tf = True


class TestRuntimeTF(unittest.TestCase):

    @unittest.skipIf(skip_tf, "Skipping Tensorflow related test because Tensorflow is not available")
    def test_split_tuple_get_item(self):
        xlayers = [
            xlayer.XLayer(
                name='in1',
                type=['Input'],
                shapes=[1, 5, 1, 1],
                sizes=[5],
                bottoms=[],
                tops=['split1'],
                attrs={},
                targets=[]
            ),
            xlayer.XLayer(
                type=['Split'],
                name='split1',
                shapes=[[1, 1, 1, 1], [1, 3, 1, 1], [1, 1, 1, 1]],
                sizes=[1, 3, 1],
                bottoms=['in1'],
                tops=[],
                targets=[],
                attrs={'axis': 1, 'indices': [1, 4]}
            ),
            xlayer.XLayer(
                type=['TupleGetItem'],
                name='tgi1',
                shapes=[1, 3, 1, 1],
                sizes=[1],
                bottoms=['split1'],
                tops=[],
                targets=[],
                attrs={'index': 1}
            ),
            xlayer.XLayer(
                type=['TupleGetItem'],
                name='tgi2',
                shapes=[1, 1, 1, 1],
                sizes=[1],
                bottoms=['split1'],
                tops=[],
                targets=[],
                attrs={'index': 2}
            )
        ]
        
        xgraph = XGraphFactory().build_from_xlayer(xlayers)
        runtime_tf = RuntimeTF('test', xgraph)

        inputs = {
            'in1': np.array([1, 2, 3, 4, 5], dtype=np.float32)
            .reshape(1, 5, 1, 1)
        }
        outpt_1, outpt_2 = runtime_tf.run(inputs, ['tgi1', 'tgi2'])

        expected_outpt_1 = np.array([2, 3, 4], dtype=np.float32)\
            .reshape(1, 3, 1, 1)
        expected_outpt_2 = np.array([5], dtype=np.float32).reshape(1, 1, 1, 1)

        np.testing.assert_array_almost_equal(outpt_1, expected_outpt_1)
        np.testing.assert_array_almost_equal(outpt_2, expected_outpt_2)

    @unittest.skipIf(skip_tf, "Skipping Tensorflow related test because Tensorflow is not available")
    def test_batch_norm(self):
        M = np.array([0.5, 1.2], dtype=np.float32)
        V = np.array([0.1, 0.05], dtype=np.float32)
        G = np.array([1.0, 1.0], dtype=np.float32)
        B = np.array([0., 0.], dtype=np.float32)

        xlayers = [xlayer.XLayer(
                name='input',
                type=['Input'],
                shapes=[1, 2, 1, 1],
                sizes=[2],
                bottoms=[],
                tops=[],
                attrs={},
                targets=[]
            ),
            xlayer.XLayer(
                name='bn',
                type=['BatchNorm'],
                shapes=[1, 2, 1, 1],
                sizes=[2],
                bottoms=['input'],
                tops=[],
                data=xlayer.BatchData(M, V, G, B),
                attrs={
                    'axis': 1,
                    'epsilon': 0.000001
                },
                targets=[]
            )]

        xgraph = XGraphFactory().build_from_xlayer(xlayers)
        runtime_tf = RuntimeTF('test', xgraph)

        inputs = {
            'input': np.array([1, 1], dtype=np.float32).reshape(1, 2, 1, 1)
        }
        outpt = runtime_tf.run(inputs)[0]

        expected_outpt = (inputs['input'] - np.reshape(M, (1, 2, 1, 1))) /\
            np.sqrt(np.reshape(V, (1, 2, 1, 1)) + 0.000001)

        np.testing.assert_array_almost_equal(outpt, expected_outpt)


if __name__ == '__main__':
    unittest.main()
