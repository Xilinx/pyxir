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

"""
Module for building xgraph for for quantization simulation cpu execution
with clipping of 12 most significant bits
"""

import os
import copy
import logging
import warnings

import pyxir

from pyxir.quantization.simulation.quant_sim_pass import XGraphQuantSimPass

# ! Override Convolution quantization simulation function
from . import qsim_12_msbits_transforms

from pyxir.graph.optimization.optimizers.q_optimizer import QOptimizer
from pyxir.quantization.default_quantizer import XGraphDefaultQuantizer
from pyxir.quantization.mse_quantization.mse_threshold_quantizer import\
    XGraphMSEThresholdQuantizer

from pyxir.graph.transformers.layout_transformation_pass \
    import XGraphLayoutTransformationPass

logger = logging.getLogger('pyxir')


def build_for_quantization_simulation(xgraph,
                                      work_dir=os.path.join(os.getcwd(),
                                                            'work'),
                                      subgraphs_only=True,
                                      **kwargs):
    # type: (XGraph, str, bool, dict) -> XGraph
    """ XGraph build function for quantization simulation """

    quant_sim_pass = XGraphQuantSimPass(
        fdir=work_dir,
        name=xgraph.get_name() + '_qsim',
        output_png='qsim_graph.png'
        if logger.getEffectiveLevel() <= 10 else None
    )
    qsim_xgraph = quant_sim_pass.execute(xgraph=xgraph,
                                         subgraphs_only=subgraphs_only)

    return qsim_xgraph


def qsim_xgraph_optimizer(xgraph, target=None, **kwargs):
    # type: (XGraph, str, dict) -> XGraph
    """ Basic xgraph optimizer """

    optimizer = QOptimizer(xgraph)
    optimizer.optimize()

    return xgraph


def qsim_xgraph_quantizer(xgraph,
                          inputs_func,
                          work_dir=os.path.join(os.getcwd(), 'work'),
                          **kwargs):
    # type: (XGraph, function, str, dict) -> XGraph
    """ Basic xgraph quantizer """

    quantizer = XGraphDefaultQuantizer(xgraph, inputs_func,
                                       work_dir=work_dir,
                                       **kwargs)
    q_xgraph = quantizer.quantize(subgraphs_only=False)

    # quantizer = XGraphMSEThresholdQuantizer(xgraph, inputs_func,
    #                                         work_dir=work_dir, **kwargs)
    # q_xgraph = quantizer.quantize(subgraphs_only=False)

    return q_xgraph


def qsim_xgraph_compiler(xgraph, **kwargs):
    # type: (XGraph) -> XGraph
    """ Basic xgraph quantizer """
    warnings.warn("'qsim' compilation just returns the original XGraph")

    return xgraph


pyxir.register_target('qsim-12msbs',
                      qsim_xgraph_optimizer,
                      qsim_xgraph_quantizer,
                      qsim_xgraph_compiler,
                      build_for_quantization_simulation)


@pyxir.register_op_support_check('qsim-12msbs', 'All')
def qsim_op_support_check(X, bXs, tXs):
    """ Enable all operations """
    return True
