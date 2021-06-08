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

import logging

from pyxir import graph
from pyxir.shared import fancy_logging
from pyxir.graph.layer import xlayer_factory
from pyxir.graph.transformers.remove_unused_ops import RemoveUnusedOps
from pyxir.graph.optimization.optimizers.transposes_optimizer import (
    XGraphTransposesOptimizer,
)
from pyxir.graph.transformers.layout_transformation_pass import (
    XGraphLayoutTransformationPass,
)

from .util import Schedule
from ..base import BaseConverter
from .relay_2_xlayer_registry import Relay2XLayerRegistry

fancy_logger = fancy_logging.getLogger("pyxir")
logger = logging.getLogger("pyxir")


class Relay2XGraphConverter(BaseConverter):

    """Class for converting Relay functions to a XGraph representation"""

    RELAY_2_XLAYER = Relay2XLayerRegistry()

    def from_relay_to_xgraph(
        self, sym, params, output_op=None, postprocessing=None, cvx_preprocessing=None
    ):
        # type: (tvm.relay.expr.Expr, dict, str, str, list, dict) ->  XGraph
        """
        Transform a TVM Relay expression to a xfDNN graph and schedule

        Arguments
        ---------
        sym: tvm.relay.expr.Expr
            the Relay expression
        params: dict
            the parameters of the Relay expression
        input_layouts: List[str] # TODO
            the layouts of the data inputs
        output_op: str
            the output operation (unused)
        postprocessing: List[str]
            list of postprocessing layers to be added
        cvx_preprocessing: Dict
            dictionary mapping input names to their cvx preprocessing
            key

        Returns:
        --------
        xgraph: XGraph
            the graph data structure containing all information
        """

        if postprocessing is None:
            postprocessing = []
        if cvx_preprocessing is None:
            cvx_preprocessing = {}

        if output_op is not None:
            raise NotImplementedError("'output_op' should be None for now")

        fancy_logger.banner("RELAY IR TO PYXIR")

        # schedule = []
        net = {}
        schedule = Schedule(net)
        # CONVERT RELAY EXPRESSION TO XLAYER GRAPH
        # This starts a rescursive expression to graph conversion function
        X = Relay2XGraphConverter.RELAY_2_XLAYER[sym.__class__.__name__](
            sym,
            params,
            schedule,
            net,
            {},
            Relay2XGraphConverter.RELAY_2_XLAYER,
            cvx_prep=cvx_preprocessing,
        )

        # For now only softmax layers can be added to a graph output
        OP_2_XLAYER = {
            "Softmax": xlayer_factory.get_xop_factory_func("Softmax", internal=True)
        }

        # Add additional output layers to the network that are not specified
        #   in the network file (usually only used for adding softmax layers)
        for i, output in enumerate(postprocessing):
            if output not in OP_2_XLAYER:
                continue

            op_name = output + str(i)

            # Update tops of current last layer
            X.tops.append(op_name)
            X = OP_2_XLAYER[output](op_name, [X])

            if X.name in net:
                raise ValueError(
                    "This should never happen. Error because the"
                    " generated output name already exists in the"
                    " network dictionary used for setup."
                )

            schedule.append(X.name)
            net[X.name] = X

        # Possibly replace Input layers with CvxInput layers
        xlayers = [net[op_id] for op_id in schedule]
        xgraph = self.xgraph_factory.build_from_xlayer(
            net=xlayers, name="relay_xgraph", blobs=False
        )

        # TODO remove this layout transformer
        layout_transform_pass = XGraphLayoutTransformationPass("NCHW")
        xgraph = layout_transform_pass.execute(xgraph, subgraphs_only=False)

        # Merge transpose layers
        t_optimizer = XGraphTransposesOptimizer(xgraph)
        t_optimizer.optimize()

        # Remove unused ops
        xgraph = RemoveUnusedOps()(xgraph)

        return xgraph
