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
Module for registering DPUV2 target


"""

import os
import json
import logging
import warnings
import numpy as np

import pyxir

from pyxir.runtime import base
from pyxir.runtime.rt_layer import BaseLayer

logger = logging.getLogger('pyxir')


class DPUV2Layer(BaseLayer):
    try:
        from dnndk import n2cube
    except:
        warnings.warn("Could not import dnndk n2cube module")

    def init(self):
        # Setup
        input_names = self.attrs['input_names']
        assert(len(input_names) == 1)
        output_names = self.attrs['output_names']
        assert(len(output_names) >= 1)

        logger.debug("SHAPE: {}".format(self.shape))

        # TODO: needed for now because DNNC compiled model input and output
        #   name are different from the input/output names provided to DNNC
        dnnc_comp_file = os.path.join(os.getcwd(),
                                      "dnnc_comp_{}.json".format(self.name))
        if not os.path.isfile(dnnc_comp_file):
            raise ValueError("Couldn't find expected file DNNC compatibility"
                             " file: {}".format(dnnc_comp_file))
        with open(dnnc_comp_file) as jf:
            d = json.load(jf)
            self.input_names = [d[in_name] for in_name in input_names]
            self.output_names = [d[out_name] for out_name in output_names]

        # Attach to DPU driver and prepare for runing
        self.n2cube.dpuOpen()

        # Create DPU Kernel
        self.kernel = self.n2cube.dpuLoadKernel(self.name)
        self.task = self.n2cube.dpuCreateTask(self.kernel, 0)

    def forward_exec(self, inputs):
        # type: (List[numpy.ndarray]) -> numpy.ndarray

        # For now
        assert(len(inputs) == 1)
        # DPU layer expects one image !!
        assert(inputs[0].shape[0] == 1)

        X = inputs[0][0].reshape((-1))
        X_len = len(X)

        # Load image to DPU in (CHW or HWC format)
        self.n2cube.dpuSetInputTensorInHWCFP32(
            self.task, self.input_names[0], X, X_len)

        # Model run on DPU
        self.n2cube.dpuRunTask(self.task)

        res = []
        for idx, out_name in enumerate(self.output_names):
            # Get the output tensor size
            size = self.n2cube.dpuGetOutputTensorSize(self.task, out_name)

            out = [0 for i in range(size)]

            # Get DPU result
            conf = self.n2cube.dpuGetOutputTensorAddress(self.task, out_name)
            self.n2cube.dpuGetTensorData(conf, out, size)

            # Get output scale
            output_scale = self.n2cube.dpuGetOutputTensorScale(self.task,
                                                               out_name)

            # TODO

            newshape = [(dim if dim is not None else -1)
                        for dim in self.shape[idx]]
            out = np.reshape(np.array(out), newshape) * float(output_scale)
            logger.debug("Out {} shape: {}".format(out_name, out.shape))
            # logger.debug("Out", out)
            res.append(out)

        # This DPULayer returns a tuple of outputs
        return tuple(res)

    def __del__(self):
        """
        Cleanup DPU resources
        """
        # Destroy DPU Tasks & free resources
        rtn = self.n2cube.dpuDestroyKernel(self.kernel)

        # Dettach from DPU driver & release resources
        self.n2cube.dpuClose()

pyxir.register_op('cpu-np', 'DPUV2', base.get_layer(DPUV2Layer))
