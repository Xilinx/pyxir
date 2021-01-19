import pyxir

from .components.DPUCZDX8G.common import xgraph_dpu_external_quantizer
from .components.DPUCZDX8G.common import xgraph_dpu_external_quantizer_optimizer
from .components.DPUCZDX8G.u50 import xgraph_dpu_u50_build_func
from .components.DPUCZDX8G.u50 import xgraph_dpu_u50_compiler


# Register target
pyxir.register_target('DPUCAHX8H-u50',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_u50_compiler,
                      xgraph_dpu_u50_build_func)

# Register op support
from .components.DPUCAHX8H import u50_op_support
