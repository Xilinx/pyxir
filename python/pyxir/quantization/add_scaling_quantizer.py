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
Module for quantizing XGraph models with additional scaling layers added
before eltwise operations


"""

import os
import logging

from pyxir.shared import QuantParams
from .quant_scaling_pass import XGraphQuantScalingPass

from .default_quantizer import XGraphDefaultQuantizer
from .quant_ops import EltwiseQuantWithScale, ConcatQuantWithScale

logger = logging.getLogger("pyxir")


class XGraphAddScalingQuantizer(XGraphDefaultQuantizer):

    """
    TODO
    """

    def __init__(self, skip_quant_compute=False, **kwargs):
        super(XGraphAddScalingQuantizer, self).__init__(**kwargs)

        # Replace quantization function for eltwise operation
        self.XFDNN_OP_2_QUANT_FUNC['Eltwise'] = \
            EltwiseQuantWithScale(self._quant_param,
                                  self._quant_layers,
                                  self._bitwidth)
        self.XFDNN_OP_2_QUANT_FUNC['Concat'] = \
            ConcatQuantWithScale(self._quant_param,
                                 self._quant_layers,
                                 self._bitwidth)

        self.skip_quant_compute = skip_quant_compute

    def quantize(self, inputs=None, stop=None, subgraphs_only=True):
        # (numpy.ndarray, str) -> None
        """
        Overrides parent method
        """
        # TODO: check valid quant file
        if not self.skip_quant_compute:
            quant_files = super(XGraphAddScalingQuantizer, self)\
                .quantize(inputs=inputs,
                          stop=stop,
                          subgraphs_only=subgraphs_only)
        else:
            quant_files = {
                xp_name: os.path.join(self.outdir, xp_name + '_quant.json')
                for xp_name in self.xgraph.get_subgraph_names()
            }

        assert(len(quant_files) == 1)

        xp_key = list(quant_files.keys())[0]
        quant_params = QuantParams(quant_files[xp_key])
        graph_pass = XGraphQuantScalingPass(
            quant_params,
            quant_files[xp_key],
            output_png='tvm_quant_eltwise_scaling.png'
            if logger.getEffectiveLevel() <= 10 else None
        )
        xgraph = graph_pass.execute(self.xgraph)

        return xgraph
