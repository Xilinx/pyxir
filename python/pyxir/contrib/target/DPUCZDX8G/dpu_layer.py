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

""" Module for registering Numpy DPU runtime layer """

import os
import json
import logging
import warnings
import numpy as np

import pyxir

from pyxir.runtime import base
from pyxir.runtime.rt_layer import BaseLayer

logger = logging.getLogger('pyxir')


class DPULayer(BaseLayer):

    try:
        from pyxir.contrib.vai_runtime.runner import Runner
    except:
        warnings.warn("Could not import Vitis-AI Runner")

    def init(self):
        # Setup
        input_names = self.attrs['input_names']
        assert(len(input_names) == 1)

        output_names = self.attrs['output_names']
        assert(len(output_names) >= 1)

        self.runner = self.Runner(self.attrs['work_dir'])

        logger.debug("SHAPE: {}".format(self.shape))

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        # For now
        assert(len(inputs) == 1)
        assert(inputs[0].shape[0] == 1)

        X = inputs[0]
        res = []
        inTensors = self.runner.get_input_tensors()
        outTensors = self.runner.get_output_tensors()

        batch_sz = 1

        fpgaBlobs = []
        for io in [inTensors, outTensors]:
            blobs = []
            for t in io:
                shape = (batch_sz,) + tuple([t.dims[i] for i in range(t.ndims)][1:])
                blobs.append(np.empty((shape), dtype=np.float32, order='C'))
            fpgaBlobs.append(blobs)

        fpgaInput = fpgaBlobs[0][0]
        np.copyto(fpgaInput[0], X[0])

        jid = self.runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])

        self.runner.wait(jid)

        res.append(fpgaBlobs[1][0])

        return tuple(res)

    def __del__(self):
        """
        Cleanup DPU resources
        """
        del self.runner

pyxir.register_op('cpu-np', 'DPU', base.get_layer(DPULayer))
