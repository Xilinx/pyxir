import pyxir

from .components.DPUCZDX8G.external_quantizer_tools import xgraph_dpu_external_quantizer
from .components.DPUCZDX8G.external_quantizer_tools import xgraph_dpu_external_quantizer_optimizer
from .components.DPUCAHX8H.u50 import xgraph_dpu_u50_build_func
from .components.DPUCAHX8H.u50 import xgraph_dpu_u50_compiler
from .components.DPUCAHX8H.u55c import xgraph_dpu_u55c_build_func
from .components.DPUCAHX8H.u55c import xgraph_dpu_u55c_compiler
from .components.DPUCAHX8H.u55c_dwc import xgraph_dpu_u55c_dwc_build_func
from .components.DPUCAHX8H.u55c_dwc import xgraph_dpu_u55c_dwc_compiler
from .components.DPUCAHX8H.u280 import xgraph_dpu_u280_build_func
from .components.DPUCAHX8H.u280 import xgraph_dpu_u280_compiler


# Register target
pyxir.register_target('DPUCAHX8H-u50',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_u50_compiler,
                      xgraph_dpu_u50_build_func)

# Register op support
from .components.DPUCAHX8H import u50_op_support

# Register target
pyxir.register_target('DPUCAHX8H-u55c',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_u55c_compiler,
                      xgraph_dpu_u55c_build_func)

# Register op support
from .components.DPUCAHX8H import u55c_op_support

# Register target
pyxir.register_target('DPUCAHX8H-u55c_dwc',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_u55c_dwc_compiler,
                      xgraph_dpu_u55c_dwc_build_func)


# Register op support
from .components.DPUCAHX8H import u55c_dwc__op_support


# Register U280 target
pyxir.register_target('DPUCAHX8H-u280',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_u280_compiler,
                      xgraph_dpu_u280_build_func)

# Register op support
from .components.DPUCAHX8H import u280_op_support
