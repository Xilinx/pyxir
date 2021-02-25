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
""""Register DPUCZDX8G targets"""

import pyxir

from .components.DPUCZDX8G.common import xgraph_dpu_quantizer
from .components.DPUCZDX8G.common import xgraph_dpu_optimizer
from .components.DPUCZDX8G.common import xgraph_dpu_op_support_annotator
from .components.DPUCZDX8G.ultra96 import xgraph_dpu_ultra96_build_func
from .components.DPUCZDX8G.ultra96 import xgraph_dpu_ultra96_compiler
from .components.DPUCZDX8G.zcu102 import xgraph_dpu_zcu102_build_func
from .components.DPUCZDX8G.zcu102 import xgraph_dpu_zcu102_compiler
from .components.DPUCZDX8G.zcu104 import xgraph_dpu_zcu104_build_func
from .components.DPUCZDX8G.zcu104 import xgraph_dpu_zcu104_compiler
from .components.DPUCZDX8G.som import xgraph_dpu_som_build_func
from .components.DPUCZDX8G.som import xgraph_dpu_som_compiler


# Register target
pyxir.register_target('DPUCZDX8G-ultra96',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_ultra96_compiler,
                      xgraph_dpu_ultra96_build_func)

# Register op support
from .components.DPUCZDX8G import ultra96_op_support

# Register target
pyxir.register_target('DPUCZDX8G-zcu102',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_zcu102_compiler,
                      xgraph_dpu_zcu102_build_func)

# Register op support
from .components.DPUCZDX8G import zcu102_op_support

# Register target
pyxir.register_target('DPUCZDX8G-zcu104',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_zcu104_compiler,
                      xgraph_dpu_zcu104_build_func,
                      xgraph_dpu_op_support_annotator)

# Register op support
from .components.DPUCZDX8G import zcu104_op_support

# Register target
pyxir.register_target('DPUCZDX8G-som',
                      xgraph_dpu_optimizer,
                      xgraph_dpu_quantizer,
                      xgraph_dpu_som_compiler,
                      xgraph_dpu_som_build_func,
                      xgraph_dpu_op_support_annotator)

# Register op support
from .components.DPUCZDX8G import som_op_support
