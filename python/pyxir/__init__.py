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

"""The PyXIR module"""

import os

from . import __import
from .base import *
from .runtime import rt_manager
from .graph import ops
from .target_registry import TargetRegistry, register_op_support_check
from pyxir.targets.cpu import build_for_cpu_execution,\
    cpu_xgraph_optimizer, cpu_xgraph_quantizer, cpu_xgraph_compiler
from .graph.xop_registry import XOpRegistry, xop_register_op_layout_transform,\
    xop_register_op_transpose_transform

__version__ = "0.2.0"


device_r = TargetRegistry()

########
# APIs #
########


def get_include_dir():
    prod_include_dir = \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")
    dev_include_dir = \
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../include")
    if os.path.exists(prod_include_dir):
        return prod_include_dir
    return dev_include_dir


def get_lib_dir():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(file_dir, '../')
    return os.path.abspath(lib_dir)


def register_rt(rt_name, rt_graph, rt_ops):
    # type: (str, BaseRuntime, dict) -> None
    rt_manager.register_rt(rt_name, rt_graph, rt_ops)


def register_op(rt_name, op_type, setup_func):
    # type: (str, str, Function) -> None
    rt_manager.register_op(rt_name, op_type, setup_func)


def register_target(target,
                    xgraph_optimizer,
                    xgraph_quantizer,
                    xgraph_compiler,
                    xgraph_build_func,
                    xgraph_op_support_annotator=None,
                    skip_if_exists=False):
    # type: (str, Function, Function, Function, Function,
    #        boolean) -> None
    device_r.register_target(target,
                             xgraph_optimizer,
                             xgraph_quantizer,
                             xgraph_compiler,
                             xgraph_build_func,
                             xgraph_op_support_annotator=xgraph_op_support_annotator,
                             skip_if_exists=skip_if_exists)

register_target('cpu',
                cpu_xgraph_optimizer,
                cpu_xgraph_quantizer,
                cpu_xgraph_compiler,
                build_for_cpu_execution)


@register_op_support_check('cpu', 'All')
def cpu_op_support_check(X, bXs, tXs):
    """ Enable all operations """
    return True


def register_op_layout_transform(xop_name):
    return xop_register_op_layout_transform(xop_name)


def register_op_transpose_transform(xop_name):
    return xop_register_op_transpose_transform(xop_name)
