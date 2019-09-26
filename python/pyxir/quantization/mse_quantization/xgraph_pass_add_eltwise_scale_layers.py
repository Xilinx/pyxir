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
Module for adding additional scaling layers after eltwise additions


"""

import numpy as np

import logging

from pyxir.shared import fancy_logging

from pyxir.shared import QuantParamFactory
from pyxir.graph.passing.base_pass import XGraphBasePass

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


def get_eltwise_parameters_layers(bottom_Ps, P, top_Ps, params):
    # type: (Dict[str, XLayer], XLayer, Dict[str, XLayer], dict) 
    #   -> List[XLayer]
    """
    TODO: Make more modular
    """
    logger.debug("Add scaling layer after eltwise: {}".format(P.name))

    new_Ps = [P]

    assert(len(bottom_Ps) == 2)

    # TODO: we are assuming a certain structure here: NCHW
    #   we have to solve this better intrinscally
    # TODO: if FPGA supports scaling layer with one scaling value
    #   we can just pass one value here = TEST
    channels = 1

    assert(len(P.shapes) == 4)
    channels = P.shapes[1]  # NCHW

    """
    gamma_name = P.name + '_gamma'
    gamma_var_attrs = {
        'init_value': np.array([1]*channels),
        'dtype': 'float32'
    }
    gamma_var_layer = XLayer(
        *[None for i in XLayer._fields])
    gamma_var_layer = gamma_var_layer._replace(
        type = ['Variable'],
        name = gamma_name,
        shapes = [1],
        attrs = gamma_var_attrs,
        bottoms = [],
        tops = []
    )
    new_Ps.append(gamma_var_layer)

    beta_name = P.name + '_beta'
    beta_var_attrs = {
        'init_value': np.array([0]*channels),
        'dtype': 'float32'
    }
    beta_var_layer = XLayer(
        *[None for i in XLayer._fields])
    beta_var_layer = beta_var_layer._replace(
        type = ['Variable'],
        name = beta_name,
        shapes = [1],
        attrs = beta_var_attrs,
        bottoms = [],
        tops = []
    )
    new_Ps.append(beta_var_layer)
    """
    gamma_name = P.name + '_gamma'
    gamma = params[gamma_name] if gamma_name in params \
        else np.array([1]*channels)

    beta_name = P.name + '_beta'
    beta = params[beta_name] if beta_name in params \
        else np.array([0]*channels)

    scale_data = xlayer.ScaleData(gamma=gamma, beta=beta)

    # Add a scaling layer
    scale_name = P.name + '_scale'
    scale_P = P._replace(
        name=scale_name,  # TODO: shapes, sizes
        type=['Scale'],
        shapes=P.shapes,
        sizes=P.sizes,
        layer=[],
        tops=[],
        bottoms=[P.name],
        data=scale_data,
        attrs={}
    )
    new_Ps.append(scale_P)

    return new_Ps


# TODO: XGraphQuantPass
class XGraphPassAddEltwiseScaleLayers(XGraphBasePass):

    """
    Responsible for inserting scaling layers for improving accuracy of
    MSE quantization

    Attributes
    ----------

    """

    def __init__(self,
                 params,
                 quant_params,
                 quantizecfg,
                 last_opt_layer=None,
                 name='XGraphQuantPass',
                 output_png=None):
        super(XGraphPassAddEltwiseScaleLayers, self).__init__(
            name=name,
            output_png=output_png)

        self.params = params
        self.quant_params = quant_params
        self.quantizecfg = quantizecfg
        self.last_opt_layer = last_opt_layer

    def execute(self, xgraph):
        # type: (XGraph) -> XGraph
        """
        """

        params = self.params
        quant_params = self.quant_params
        quant_param_factory = QuantParamFactory()

        skip = False

        def add_scaling_layer_after_eltwise(bottom_Ps, P, top_Ps):
            # type: (Dict[str, XLayer], XLayer,
            #  Dict[str, XLayer]) -> List[XLayer]
            """
            Replace the provided parameters layer with a list of parameter
            layers adding scaling layers before elementwise additions
            """
            nonlocal skip, params, quant_params

            new_Ps = []

            if 'Eltwise' in P.type and P.name in quant_params:
                new_eltwise_layers = \
                    get_eltwise_parameters_layers(bottom_Ps, P, top_Ps, params)
                new_Ps.extend(new_eltwise_layers)
                scale_name = new_Ps[-1].name
                channels = new_Ps[-1].shapes[1]  # NCHW

                logger.debug("channels", channels)

                th_layer_out = quant_params[P.name]['th_layer_out']
                bitwidth = quant_params[P.name]['bw_layer_in']

                qp = quant_param_factory.get_default_quant_params(
                    scale_name, bitwidth, channels, th_layer_out, th_layer_out)
                # logger.debug("qp", qp)

                # TODO: replace or insert
                quant_params.insert_with_replace(scale_name, qp, P.name)

            else:
                P = P._replace(
                    tops=[]  # important
                )
                new_Ps.append(P)

            # TODO merge skipping fuctionality as it is common in a lot of
            #   graph passes
            if self.last_opt_layer is not None and\
                    self.last_opt_layer == P.name:
                # Start skipping adding threshold optimization layers from the
                #   next layer
                skip = True

            return new_Ps

        # 1.
        fancy_logger.banner("GRAPH PASS ADD ELTWISE SCALE LAYERS")

        output_png = None if self.output_png is None else \
            self.output_png.split('.')[0] + '_add.' + \
            self.output_png.split('.')[1]
        xgraph = self._replace_layer_pass(
            xgraph,
            add_scaling_layer_after_eltwise,
            name=self.name + "_add_eltwise_scale_layers",
            output_png=output_png
        )

        # Save the adjusted quantization parameters
        quant_params.save(self.quantizecfg)

        return xgraph
