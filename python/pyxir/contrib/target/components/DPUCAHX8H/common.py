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

""" Module for registering common functionality"""

import logging

from pyxir.generator.tensorflow import XGraphTfGeneratorOptimizer
from pyxir.graph.optimization.optimizers import ExternalQOptimizer
from pyxir.graph.transformers.layout_transformation_pass import XGraphLayoutTransformationPass
from pyxir.quantization.decent_quantizer import DECENTQuantizer
from pyxir.quantization.external_quantizer import ExternalQuantizerDecentOutput

logger = logging.getLogger('pyxir')


def xgraph_dpu_optimizer(xgraph, target=None, **kwargs):
    layout_transform_pass = XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)
    optimizer = XGraphTfGeneratorOptimizer(dpu_xgraph)
    optimizer.optimize()
    return dpu_xgraph


def xgraph_dpu_quantizer(xgraph, inputs_func, **kwargs):
    quantizer = DECENTQuantizer(xgraph, inputs_func, **kwargs)
    q_xgraph = quantizer.quantize()
    return q_xgraph


def xgraph_dpu_external_quantizer_optimizer(xgraph, target=None, **kwargs):
    layout_transform_pass = XGraphLayoutTransformationPass('NHWC', target=target)
    dpu_xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)
    optimizer = ExternalQOptimizer(dpu_xgraph)
    optimizer.optimize()
    return dpu_xgraph


def xgraph_dpu_external_quantizer(xgraph, inputs_func, **kwargs):
    quantizer = ExternalQuantizerDecentOutput(xgraph, inputs_func, **kwargs)
    q_xgraph = quantizer.quantize()
    return q_xgraph
