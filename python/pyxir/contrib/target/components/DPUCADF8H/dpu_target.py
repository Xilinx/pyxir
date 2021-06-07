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
Module for registering DPUCADF8H target and corresponding graph optimizer,
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
from pyxir.graph.transformers.layout_transformation_pass import \
    XGraphLayoutTransformationPass
from pyxir.quantization.decent_quantizer import DECENTQuantizer
from .vai_c import VAICompiler


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


def xgraph_dpu_build_func(xgraph, work_dir=os.getcwd(), data_layout='NHWC', **kwargs) -> XGraph:
    """
    Build/schedule and XGraph for execution on the DPUCADF8H target

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
    An XGraph built/scheduled for execution on DPU
    """
    # NOTE DPUCADF8H layers are in NHWC format because of the tensorflow
    #   intemediate structure we use to communicate with  DPUCADF8H compiler
    return subgraph.xgraph_build_func(
        xgraph=xgraph,
        target='DPUCADF8H',
        xtype='DPU',
        layout=data_layout,
        work_dir=work_dir
    )


def xgraph_dpu_optimizer(xgraph, target=None, **kwargs):
    """Optimize/transform XGraph for execution on this DPU"""
    
    # Annotate and merge patterns (e.g. mul + max = leaky relu)
    XGraphPatternAnnotator()(xgraph)
    xgraph = XGraphPatternMutator()(xgraph)

    layout_transform_pass = \
        XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)

    optimizer = XGraphTfGeneratorOptimizer(dpu_xgraph)
    optimizer.optimize()

    return dpu_xgraph


def xgraph_dpu_quantizer(xgraph, inputs_func, **kwargs):
    """Quantize XGraph for execution on this DPU"""
    quantizer = DECENTQuantizer(xgraph, inputs_func, compiler_target='xcompiler', **kwargs)
    q_xgraph = quantizer.quantize()

    return q_xgraph


def xgraph_dpu_compiler(xgraph, **kwargs):
    """The DPU specific compiler function"""

    # Vitis-AI 1.3 - ...
    arch = "/opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json" 
    compiler = VAICompiler(xgraph, arch=arch, **kwargs)
    c_xgraph = compiler.compile()

    return c_xgraph
