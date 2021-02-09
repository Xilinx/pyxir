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

import pyxir
from pyxir.runtime import base

from .components.DPUCADX8G.dpu_target import xgraph_dpu_build_func
from .components.DPUCADX8G.dpu_target import xgraph_dpu_compiler
from .components.DPUCADX8G.dpu_target import DPULayer
from .components.DPUCADX8G.external_quantizer_tools import xgraph_dpu_external_quantizer
from .components.DPUCADX8G.external_quantizer_tools import xgraph_dpu_external_quantizer_optimizer


# Register target
pyxir.register_target('DPUCADX8G',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_compiler,
                      xgraph_dpu_build_func)

# Register layer
pyxir.register_op('cpu-np', 'DPU', base.get_layer(DPULayer))

# Register op support
from .components.DPUCADX8G import dpu_op_support