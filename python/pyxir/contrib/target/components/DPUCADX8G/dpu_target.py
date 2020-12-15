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
Module for registering DPUCADX8G target and corresponding graph optimizer,
quantizer, compiler and build function
"""

import os
import logging
import warnings
import numpy as np

import pyxir
from pyxir.runtime import base
from pyxir.runtime.rt_layer import BaseLayer
from pyxir.graph.transformers import subgraph
from pyxir.generator.tensorflow import XGraphTfGeneratorOptimizer
from pyxir.graph.optimization.optimizers import QOptimizer, ExternalQOptimizer
from pyxir.quantization.default_quantizer import XGraphDefaultQuantizer
from pyxir.quantization.mse_quantization.mse_threshold_quantizer import\
    XGraphMSEThresholdQuantizer
from pyxir.quantization.external_quantizer import ExternalQuantizerTxtOutput
from pyxir.graph.transformers.layout_transformation_pass import \
    XGraphLayoutTransformationPass
from pyxir.quantization.decent_quantizer import DECENTQuantizer
from .dpu_compiler import DPUCompiler

logger = logging.getLogger('pyxir')

# TARGET #

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def xgraph_dpu_build_func(xgraph, work_dir=os.getcwd(), **kwargs):

    # NOTE DPU V1 layers are in NHWC format because of the tensorflow
    #   intemediate structure we use to communicate with dpu v1 compiler
    return subgraph.xgraph_build_func(
        xgraph=xgraph,
        target='DPUCADX8G',
        xtype='DPU',
        layout='NCHW',
        work_dir=work_dir
    )


def xgraph_dpu_optimizer(xgraph, target=None, **kwargs):

    layout_transform_pass = \
        XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)

    # optimizer = QOptimizer(dpu_xgraph)
    # optimizer.optimize()
    optimizer = XGraphTfGeneratorOptimizer(dpu_xgraph)
    optimizer.optimize()

    return dpu_xgraph


def xgraph_dpu_external_quantizer_optimizer(xgraph, target=None, **kwargs):

    layout_transform_pass = \
        XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)

    optimizer = ExternalQOptimizer(dpu_xgraph)
    optimizer.optimize()

    return dpu_xgraph


def xgraph_dpu_external_quantizer(xgraph, inputs_func, **kwargs):
    quantizer = ExternalQuantizerTxtOutput(xgraph, inputs_func, **kwargs)
    q_xgraph = quantizer.quantize()
    return q_xgraph


def xgraph_dpu_quantizer(xgraph, inputs_func, **kwargs):

    # quantizer = XGraphDefaultQuantizer(xgraph, inputs_func, **kwargs)
    # q_xgraph = quantizer.quantize()

    # quantizer = XGraphMSEThresholdQuantizer(xgraph, inputs_func, **kwargs)
    # q_xgraph = quantizer.quantize()
    quantizer = DECENTQuantizer(xgraph, inputs_func, compiler_target='DPUv1Compiler', **kwargs)
    q_xgraph = quantizer.quantize()

    return q_xgraph


def xgraph_dpu_compiler(xgraph, **kwargs):

    # TODO: can we move to docker paths to arch file?
    # Vitis-AI 1.1
    old_arch = "/opt/vitis_ai/compiler/arch/dpuv1/ALVEO/ALVEO.json"
    # Vitis-AI 1.2 - ...
    new_arch = "/opt/vitis_ai/compiler/arch/DPUCADX8G/ALVEO/arch.json"

    if os.path.exists(new_arch):
        arch = os.path.join(FILE_PATH, 'arch.json')
    else:
        arch = os.path.join(FILE_PATH, 'arch_vai_11.json')
    
    compiler = DPUCompiler(xgraph, target='DPUCADX8G', arch=arch, **kwargs)
    c_xgraph = compiler.compile()

    return c_xgraph


# Register DPU numpy layer


class DPULayer(BaseLayer):

    try:
        from vai.dpuv1.rt.vitis.python.dpu.runner import Runner
    except Exception as e:
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
                shape = (batch_sz,) + tuple([t.dims[i]
                                             for i in range(t.ndims)][1:])
                blobs.append(np.empty((shape), dtype=np.float32, order='C'))
            fpgaBlobs.append(blobs)

        fpgaInput = fpgaBlobs[0][0]
        np.copyto(fpgaInput[0], X[0])

        jid = self.runner.execute_async(fpgaBlobs[0], fpgaBlobs[1])
        self.runner.wait(jid)

        res.append(fpgaBlobs[1][0])
        

        return tuple(res)

    def __del__(self):
        """ Cleanup DPU resources """
        self.runner.__del__()
