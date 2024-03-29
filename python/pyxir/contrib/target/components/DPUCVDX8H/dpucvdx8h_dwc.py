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

"""Module for registering DPUCVDX8H DWC target"""

import os
import json
import pyxir
import logging

from pyxir.generator.tensorflow import XGraphTfGeneratorOptimizer
from pyxir.graph.optimization.optimizers import ExternalQOptimizer
from pyxir.graph.transformers.layout_transformation_pass import (
    XGraphLayoutTransformationPass,
)
from pyxir.graph.transformers import subgraph
from pyxir.quantization.decent_quantizer import DECENTQuantizer
from pyxir.contrib.target.components.common.vai_c import VAICompiler


logger = logging.getLogger("pyxir")


def xgraph_dpu_optimizer_dwc(xgraph, target=None, **kwargs):
    layout_transform_pass = XGraphLayoutTransformationPass("NHWC", target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)
    optimizer = XGraphTfGeneratorOptimizer(dpu_xgraph)
    optimizer.optimize()
    return dpu_xgraph


def xgraph_dpu_quantizer_dwc(xgraph, inputs_func, **kwargs):
    quantizer = DECENTQuantizer(
        xgraph, inputs_func, compiler_target="xcompiler", **kwargs
    )
    q_xgraph = quantizer.quantize()
    return q_xgraph


def xgraph_dpu_build_func_dwc(xgraph, work_dir=os.getcwd(), **kwargs):
    return subgraph.xgraph_build_func(
        xgraph=xgraph, target="DPUCVDX8H-dwc", xtype="DPU", layout="NHWC", work_dir=work_dir
    )


def xgraph_dpu_compiler_dwc(xgraph, **kwargs):
    # Vitis-AI 2.0 - ...
    new_arch = "/opt/vitis_ai/compiler/arch/DPUCVDX8H/VCK50006PEDWC/arch.json"

    if os.path.exists(new_arch):
        arch_path = new_arch
    else:
        arch_path = None

    compiler = VAICompiler(xgraph, arch=arch_path, **kwargs)
    c_xgraph = compiler.compile()

    return c_xgraph
