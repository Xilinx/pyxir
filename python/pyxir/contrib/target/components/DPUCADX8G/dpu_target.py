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
from pyxir.graph import XGraph, XLayer
from pyxir.target import Target, DefaultOpSupportPass
from pyxir.runtime import base
from pyxir.runtime.rt_layer import BaseLayer
from pyxir.graph.transformers import subgraph
from pyxir.graph.pattern import XGraphPatternMutator, XGraphPatternAnnotator
from pyxir.generator.tensorflow import XGraphTfGeneratorOptimizer
from pyxir.graph.optimization.optimizers import QOptimizer, ExternalQOptimizer
from pyxir.quantization.default_quantizer import XGraphDefaultQuantizer
# from pyxir.quantization.mse_quantization.mse_threshold_quantizer import\
#     XGraphMSEThresholdQuantizer
from pyxir.graph.transformers.layout_transformation_pass import \
    XGraphLayoutTransformationPass
from pyxir.quantization.decent_quantizer import DECENTQuantizer
from .dpu_compiler import DPUCompiler

logger = logging.getLogger('pyxir')

# TARGET #

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

class OpSupportPass(DefaultOpSupportPass):

    def __init__(self, target: Target):
        super().__init__(target)

    def __call__(self, xg: XGraph) -> None:
        """Call Pattern Annotator pass on XGraph before calling default op support functionality"""
        XGraphPatternAnnotator()(xg)
        super(OpSupportPass, self).__call__(xg)


def xgraph_dpu_op_support_annotator(xg: XGraph, target: Target, **kwargs) -> None:
    OpSupportPass(target)(xg)


def xgraph_dpu_build_func(xgraph, work_dir=os.getcwd(), data_layout='NCHW', **kwargs) -> XGraph:
    """
    Build/schedule and XGraph for execution on the DPUCADX8G target

    Arguments:
    ----------
    xgraph: XGraph
        the xgraph to be built for execution
    work_dir: str
        the path to the work directory to be used
    data_layout: str
        the layout to be used for the DPU partitions, is NCHW by default but can be
        overridden for certain runtimes, for example the decentq simulation runtime
        makes use of this because quantization simulation is done in NHWC data layout
        instead of the NCHW data layout of the DPU

    Returns:
    --------
    And XGraph built/scheduled for execution on DPU
    """
    # NOTE DPU V1 layers are in NHWC format because of the tensorflow
    #   intemediate structure we use to communicate with dpu v1 compiler
    return subgraph.xgraph_build_func(
        xgraph=xgraph,
        target='DPUCADX8G',
        xtype='DPU',
        layout=data_layout,
        work_dir=work_dir
    )


def xgraph_dpu_optimizer(xgraph, target=None, **kwargs):
    # Annoate and merge patterns (e.g. mul + max = leaky relu)
    XGraphPatternAnnotator()(xgraph)
    xgraph = XGraphPatternMutator(xgraph)()

    layout_transform_pass = \
        XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)

    # optimizer = QOptimizer(dpu_xgraph)
    # optimizer.optimize()
    optimizer = XGraphTfGeneratorOptimizer(dpu_xgraph)
    optimizer.optimize()

    return dpu_xgraph


def xgraph_dpu_quantizer(xgraph, inputs_func, **kwargs):

    # quantizer = XGraphDefaultQuantizer(xgraph, inputs_func, **kwargs)
    # q_xgraph = quantizer.quantize()

    # quantizer = XGraphMSEThresholdQuantizer(xgraph, inputs_func, **kwargs)
    # q_xgraph = quantizer.quantize()
    quantizer = DECENTQuantizer(xgraph, inputs_func, compiler_target='DPUv1Compiler', **kwargs)
    q_xgraph = quantizer.quantize()

    return q_xgraph


def xgraph_dpu_compiler(xgraph, **kwargs):
    """The DPU specific compiler function"""
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
