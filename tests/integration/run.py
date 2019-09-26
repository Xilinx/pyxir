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
Module for running networks for testing


"""
import pyxir

from pyxir.graph.optimization.optimizers.basic_optimizer import \
    XGraphBasicOptimizer


def _run_network_cpu(xgraph, inputs, batch_size=1):
    """
    Optimize graph with using basic optimizations and run on a CPU
    """

    # Optimize graph
    optimizer = XGraphBasicOptimizer(xgraph)
    opt_xgraph = optimizer.optimize()

    # Build for CPU execution
    rt_mod = pyxir.build(opt_xgraph, target='cpu')

    # Execute
    res = pyxir.run(rt_mod, inputs, [], batch_size=batch_size)

    return res
