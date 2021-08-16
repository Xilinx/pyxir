import pyxir

from .components.DPUCZDX8G.external_quantizer_tools import xgraph_dpu_external_quantizer
from .components.DPUCZDX8G.external_quantizer_tools import xgraph_dpu_external_quantizer_optimizer
from .components.DPUCZDX8G.dpucahx8l import xgraph_dpu_build_func
from .components.DPUCZDX8G.dpucahx8l import xgraph_dpu_compiler



# Register target
pyxir.register_target('DPUCAHX8L',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_compiler,
                      xgraph_dpu_build_func)

# Register op support
from .components.DPUCAHX8L import op_support