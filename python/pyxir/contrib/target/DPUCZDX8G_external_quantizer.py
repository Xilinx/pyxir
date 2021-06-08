import pyxir

from .components.DPUCZDX8G.external_quantizer_tools import xgraph_dpu_external_quantizer
from .components.DPUCZDX8G.external_quantizer_tools import xgraph_dpu_external_quantizer_optimizer
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
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_ultra96_compiler,
                      xgraph_dpu_ultra96_build_func)

# Register op support
from .components.DPUCZDX8G import ultra96_op_support

# Register target
pyxir.register_target('DPUCZDX8G-zcu102',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_zcu102_compiler,
                      xgraph_dpu_zcu102_build_func)

# Register op support
from .components.DPUCZDX8G import zcu102_op_support

# Register target
pyxir.register_target('DPUCZDX8G-zcu104',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_zcu104_compiler,
                      xgraph_dpu_zcu104_build_func)

# Register op support
from .components.DPUCZDX8G import zcu104_op_support

# Register target
pyxir.register_target('DPUCZDX8G-som',
                      xgraph_dpu_external_quantizer_optimizer,
                      xgraph_dpu_external_quantizer,
                      xgraph_dpu_som_compiler,
                      xgraph_dpu_som_build_func)

# Register op support
from .components.DPUCZDX8G import som_op_support
