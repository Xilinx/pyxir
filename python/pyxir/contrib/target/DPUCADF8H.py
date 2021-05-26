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
""""Register DPUCADF8H target"""

import pyxir
from pyxir.runtime import base

from .components.DPUCADF8H.dpu_target import xgraph_dpu_build_func
from .components.DPUCADF8H.dpu_target import xgraph_dpu_optimizer
from .components.DPUCADF8H.dpu_target import xgraph_dpu_compiler
from .components.DPUCADF8H.dpu_target import xgraph_dpu_quantizer
from .components.DPUCADF8H.dpu_target import xgraph_dpu_op_support_annotator


# Register target
pyxir.register_target('DPUCADF8H',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_compiler,
                      xgraph_dpu_build_func,
                      xgraph_dpu_op_support_annotator)

# Register op support
from .components.DPUCADF8H import dpu_op_support
