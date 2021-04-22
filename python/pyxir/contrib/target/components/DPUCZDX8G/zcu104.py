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

""" Module for registering DPUCZDX8G zcu104 target """

import os
import json
import pyxir
import logging

from pyxir.graph.transformers import subgraph
from pyxir.contrib.target.components.common.vai_c import VAICompiler
from pyxir.contrib.target.components.common import is_dpuczdx8g_vart_flow_enabled

from .vai_c_dnnc import VAICompilerDNNC
from .common import xgraph_dpu_optimizer, xgraph_dpu_quantizer


logger = logging.getLogger('pyxir')

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
def xgraph_dpu_zcu104_build_func(xgraph, work_dir=os.getcwd(), **kwargs):

    # TODO here or in optimizer, both?
    # DPU layers are in NHWC format because of the tensorflow
    #   intemediate structure we use to communicate with
    #   DECENT/DNNC

    return subgraph.xgraph_build_func(
        xgraph=xgraph,
        target='DPUCZDX8G-zcu104',
        xtype='DPU',
        layout='NHWC',
        work_dir=work_dir
    )


def xgraph_dpu_zcu104_compiler(xgraph, **kwargs):
    if is_dpuczdx8g_vart_flow_enabled():
        arch_path = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"
        compiler = VAICompiler(xgraph, arch=arch_path, **kwargs)
        c_xgraph = compiler.compile()
    else:
        meta = {
            "lib": "/usr/local/lib/libn2cube.so",
            "pre_processing_pool": 4,
            "post_processing_pool": 4,
            "dpu_thread_pool": 3,
            "dpu_task_pool": 16
        }

        dcf_path =  os.path.join(FILE_DIR, "./ZCU104.dcf")
        arch_path = "/tmp/ZCU104.json"
        if not os.path.exists(arch_path):
            # Write arch json 
            arch = {   
                "target"   : "DPUCZDX8G",
                "dcf"      : dcf_path,
                "cpu_arch" : "arm64"
            }

            with open(arch_path, 'w') as f:
                json.dump(arch, f, indent=4, sort_keys=True)
        
        compiler = VAICompilerDNNC(xgraph, arch=arch_path, meta=meta, dcf=dcf_path, **kwargs)
        c_xgraph = compiler.compile()

    return c_xgraph
