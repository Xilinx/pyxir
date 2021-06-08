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
import pyxir

from pyxir.graph.transformers import subgraph
from pyxir.runtime import base

from ..target import DPUCADX8G
from ..target.components.DPUCADX8G.dpu_target import DPULayer
from ..target.components.DPUCADX8G.dpu_target import xgraph_dpu_optimizer
from ..target.components.DPUCADX8G.dpu_target import xgraph_dpu_quantizer
from ..target.components.DPUCADX8G.dpu_compiler import DPUCompiler

logger = logging.getLogger('pyxir')

# TARGET #

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def xgraph_dpuv1_build_func(xgraph, work_dir=os.getcwd(), **kwargs):

    # NOTE DPU V1 layers are in NHWC format because of the tensorflow
    #   intemediate structure we use to communicate with dpu v1 compiler
    return subgraph.xgraph_build_func(
        xgraph=xgraph,
        target='dpuv1',
        xtype='DPU',
        layout='NCHW',
        work_dir=work_dir
    )

def xgraph_dpuv1_compiler(xgraph, **kwargs):

    # TODO: can we move to docker paths to arch file?
    # Vitis-AI 1.1
    old_arch = "/opt/vitis_ai/compiler/arch/dpuv1/ALVEO/ALVEO.json"
    # Vitis-AI 1.2 - ...
    new_arch = "/opt/vitis_ai/compiler/arch/DPUCADX8G/ALVEO/arch.json"

    if os.path.exists(new_arch):
        arch = os.path.join(FILE_PATH, '../target/components/DPUCADX8G/arch.json')
    else:
        arch = os.path.join(FILE_PATH, '../target/components/DPUCADX8G/arch_vai_11.json')
    
    compiler = DPUCompiler(xgraph, 'dpuv1', arch, **kwargs)
    c_xgraph = compiler.compile()

    return c_xgraph


pyxir.register_target('dpuv1',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpuv1_compiler,
                      xgraph_dpuv1_build_func)


# pyxir.register_op('cpu-np', 'DPU', base.get_layer(DPULayer))
