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

""" Module for registering DPUCZDX8G u280 target """

import os
import json
import pyxir
import logging

from pyxir.graph.transformers import subgraph

from .common import xgraph_dpu_optimizer, xgraph_dpu_quantizer
from .vai_c import VAICompiler

logger = logging.getLogger('pyxir')

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
def xgraph_dpu_u280_build_func(xgraph, work_dir=os.getcwd(), **kwargs):

    # TODO here or in optimizer, both?
    # DPU layers are in NHWC format because of the tensorflow
    #   intemediate structure we use to communicate with
    #   DECENT/DNNC

    return subgraph.xgraph_build_func(
        xgraph=xgraph,
        target='DPUCAHX8L-u280',
        xtype='DPU',
        layout='NHWC',
        work_dir=work_dir
    )


def xgraph_dpu_u280_compiler(xgraph, **kwargs):

    # Vitis-AI 1.3 - ...
    arch = "/opt/vitis_ai/compiler/arch/DPUCAHX8L/U280/arch.json"
    if not os.path.isfile(arch):
        raise ValueError("Arch file: {} does not exist".format(arch))


    compiler = VAICompiler(xgraph, arch=arch, **kwargs)
    c_xgraph = compiler.compile()

    return c_xgraph
