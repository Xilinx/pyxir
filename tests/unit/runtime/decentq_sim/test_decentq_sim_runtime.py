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

"""Module for testing the pyxir Decent quantizer simulation runtime"""

import unittest
import numpy as np
import pyxir as px

from pyxir.runtime import base
from pyxir.runtime.decentq_sim.runtime_decentq_sim import RuntimeDecentQSim
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.layer import xlayer


class TestDecentQSimRuntime(unittest.TestCase):

    def test_init(self):
        K = np.reshape(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
                       (2, 1, 2, 2))

        i = px.ops.input('input', [1, 1, 4, 4])
        k = px.ops.constant('kernel', K)
        c = px.ops.conv2d(
            op_name='conv1',
            input_layer=i,
            weights_layer=k,
            kernel_size=[2, 2],
            strides=[1, 1],
            padding_hw=[0, 0],
            dilation=[1, 1],
            groups=1,
            channels=2,
            data_layout='NCHW',
            kernel_layout='OIHW'
        )
        c.target='cpu'
        c.subgraph='xp0'

        xlayers = [i, k, c]
        xgraph = XGraphFactory().build_from_xlayer(xlayers)
        xgraph.meta_attrs['quant_keys'] = ['xp0']
        xgraph.meta_attrs['xp0'] = {'q_eval': '/path/to/q_eval'}
        sim_runtime = RuntimeDecentQSim('test', xgraph)
        # We can succesfully initialize a RuntimeDecentQSim object

if __name__ == '__main__':
    unittest.main()
