# Copyright 2021 Xilinx Inc.
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
""""Register DPUCVDX8H target"""

import pyxir

from .components.DPUCVDX8H.dpucvdx8h import xgraph_dpu_quantizer
from .components.DPUCVDX8H.dpucvdx8h import xgraph_dpu_optimizer
from .components.DPUCVDX8H.dpucvdx8h import xgraph_dpu_build_func
from .components.DPUCVDX8H.dpucvdx8h import xgraph_dpu_compiler


# Register target
pyxir.register_target(
    "DPUCVDX8H",
    xgraph_dpu_optimizer,
    xgraph_dpu_quantizer,
    xgraph_dpu_compiler,
    xgraph_dpu_build_func,
)

# Register op support
from .components.DPUCVDX8H import op_support
