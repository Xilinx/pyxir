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

"""Module for quantizing xgraphs with subgraphs"""

import os
import logging
import numpy as np
import pyxir

from typing import List, Dict

from pyxir.graph import XGraph
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.quantization.base_quantizer import XGraphBaseQuantizer

logger = logging.getLogger("pyxir")


class XGraphBaseSubgraphQuantizer(XGraphBaseQuantizer):

    """
    Base class for quantization of XGraphs with partitioned subgraphs

    Attributes
    ----------
    xgraph: XGraph
        the XGraph instance to be quantized
    inputs_func: function
        the input function to be used for retrieving model inputs
    work_dir: str
        the directory to be used for writing files
    quant_iter:
        the number of quantization iterations to be done
    """

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()

    def __init__(
        self, xgraph, inputs_func, work_dir=os.path.join(os.getcwd()), quant_iter=1
    ):
        #
        super(XGraphBaseSubgraphQuantizer, self).__init__(xgraph)

        self.subgraph_Xps = XGraphBaseSubgraphQuantizer.xgraph_partitioner.get_subgraphs(
            self.xgraph
        )

        # Maps external (graph) to internal (subgraph) inputs for each subgraph
        self.subgraph_input_map = {}
        self.subgraph_inputs = {}
        self.subgraph_input_names = []
        for Xp in self.subgraph_Xps:
            sub_xgraph = XGraphBaseSubgraphQuantizer.xgraph_factory.build_from_xlayer(
                Xp.subgraph_data, name=Xp.name
            )

            self.subgraph_input_map[Xp.name] = {}

            input_names = sub_xgraph.get_input_names()

            for b, in_name in zip(Xp.bottoms, input_names):
                self.subgraph_input_names.append(b)
                self.subgraph_inputs[b] = None
                self.subgraph_input_map[Xp.name][b] = in_name

        # Setup executable graph
        self.runtime = pyxir.build(
            self.xgraph, target="cpu", last_layers=self.subgraph_input_names
        )

        self.inputs_func = inputs_func
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.quant_iter = quant_iter

    def quantize_subgraph(
        self,
        xgraph: XGraph,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
    ) -> None:
        """Quantize a subgraph with given calibration inputs"""
        raise NotImplementedError("")

    def quantize(self) -> XGraph:
        """Start quantization of the partitioned xgraph
        
        Returns
        -------
        q_xgraph: The quantized XGraph
        """

        input_names = self.runtime.get_input_names()
        input_shapes = self.runtime.get_input_shapes()
        assert len(input_names) == len(input_shapes)
        if len(input_names) != 1:
            raise NotImplementedError(
                "Invalid number of inputs to model: {},{}, Vitis-AI"
                " quantization only supports models with one input at the"
                " moment".format(len(input_names), input_names)
            )
        input_name = input_names[0]
        input_shape = input_shapes[0]

        logger.debug("START Compute subgraph inputs for quantization")
        for it in range(self.quant_iter):
            inputs = self.inputs_func(it)

            subgraph_inpts = self.runtime.run(inputs, outputs=self.subgraph_input_names)

            for in_name, inpt in zip(self.subgraph_input_names, subgraph_inpts):
                self.subgraph_inputs[in_name] = inpt

        logger.debug("START Subgraph quantization")
        for Xp in self.subgraph_Xps:
            # Create sub XGraph from subgraph layer subgraph_data
            sub_xgraph = XGraphBaseSubgraphQuantizer.xgraph_factory.build_from_xlayer(
                Xp.subgraph_data, name=Xp.name
            )

            input_names = list(Xp.attrs["__bottom_tensors"].keys())
            output_names = list(Xp.attrs["__top_tensors"].keys())

            original_input_names = list(self.subgraph_input_map[Xp.name].keys())
            inputs = {
                self.subgraph_input_map[Xp.name][in_name]: self.subgraph_inputs[in_name]
                for in_name in original_input_names
            }
            self.quantize_subgraph(sub_xgraph, inputs, input_names, output_names)

        logger.debug("STOP Subgraph quantization")

        return self.xgraph
