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

"""Module for transforming TVM Relay expression to XGraph representation"""

import tvm
import logging

from pyxir.shared import fancy_logging

from .relay_tools.relay_l0_expr_and_others import *
from .relay_tools.relay_l1_basic import *
from .relay_tools.relay_l2_convolution import *
from .relay_tools.relay_l3_math_and_transform import *
from .relay_tools.relay_l4_broadcast_and_reduction import *
from .relay_tools.relay_l5_vision import *
from .relay_tools.relay_l10_temporary import *

from .relay_tools.relay_2_xgraph_converter import Relay2XGraphConverter

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


def from_relay(sym,
               params,
               output_op=None,
               postprocessing=None,
               cvx_preprocessing=None
               ):
    # type: (tvm.relay.module.Module/tvm.relay.expr.Function, dict,
    #   str, str, str) -> XGraph
    """ Main function to import a Relay expression """
    if isinstance(sym, tvm.ir.module.IRModule):
        sym = sym.functions[sym.get_global_var('main')]
    if not isinstance(sym, tvm.relay.function.Function):
        raise ValueError("Invalid type for `sym` argument: {}, should be of"
                         " type `tvm.relay.module.Module` or "
                         " `tvm.relay.expr.Function`".format(type(sym)))

    converter = Relay2XGraphConverter()
    xgraph = converter.from_relay_to_xgraph(
        sym,
        params,
        output_op=output_op,
        postprocessing=postprocessing,
        cvx_preprocessing=cvx_preprocessing
    )

    fancy_logger.banner("GRAPH IMPORTED FROM RELAY")

    return xgraph
