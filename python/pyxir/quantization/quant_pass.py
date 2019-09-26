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
Module for passing through XGraph models for quantization related adjustments


"""

from pyxir.graph.passing.base_pass import XGraphBasePass


class XGraphQuantPass(XGraphBasePass):

    """
    Base class for quantization related graph passing

    Arguments
    ---------
    quant_params: quantize.QuantParams
        the quantization parameters that have been retrieved from quantization
    quantizecfg: str
        the path to the file to store the new quantization parameters
    name: str
        the new name of the xgraph
    output_png: str
        the name of the png file for graph visualization if specified
    """

    def __init__(self,
                 quant_params,
                 quantizecfg,
                 name='XGraphQuantPass',
                 output_png=None):
        super(XGraphQuantPass, self).__init__(name=name, output_png=output_png)

        self.quant_params = quant_params
        self.quantizecfg = quantizecfg
