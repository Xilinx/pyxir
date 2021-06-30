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
""""Register DPUCAHX8L targets"""

import pyxir

from .components.DPUCAHX8L.common import xgraph_dpu_quantizer
from .components.DPUCAHX8L.common import xgraph_dpu_optimizer
from .components.DPUCAHX8L.u50 import xgraph_dpu_u50_build_func
from .components.DPUCAHX8L.u50 import xgraph_dpu_u50_compiler
from .components.DPUCAHX8L.u280 import xgraph_dpu_u280_build_func
from .components.DPUCAHX8L.u280 import xgraph_dpu_u280_compiler



# Register target
pyxir.register_target('DPUCAHX8L-u50',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_u50_compiler,
                      xgraph_dpu_u50_build_func)

# Register op support
from .components.DPUCAHX8L import u50_op_support

# Register U280 target
pyxir.register_target('DPUCAHX8L-u280',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_u280_compiler,
                      xgraph_dpu_u280_build_func)

# Register op support
from .components.DPUCAHX8L import u280_op_support
