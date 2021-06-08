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

"""Module for quantizing XGraph models"""

import os
import numpy as np
import logging
import pyxir

from pyxir.shared import QuantParamFactory

from .base_quantizer import XGraphBaseQuantizer
from .quant_ops import SkipQuant, DefaultQuant, InputQuant, ScaleQuant, \
    EltwiseQuant, PoolQuant, ConvQuant, ConcatQuant, BatchNormQuant
from pyxir.shared.quantizer_output import QuantizerOutput

logger = logging.getLogger("pyxir")


class XGraphDefaultQuantizer(XGraphBaseQuantizer):

    """

    Attributes
    ----------
    xgraph: XGraph
        the XGraph instance to be quantized
    inputs_func: Function
        the inputs functions to be used for quantization, should accept and
        iterator and return a dictionary mapping from input names to example
        input data
    bitwidth: int
        the bitwidth to be used for quantization
    work_dir: str
        the work firectory to be used for storing quantization files
    quant_iter: int
        the number of iterations for quantization
    """

    def __init__(self,
                 xgraph,
                 inputs_func,
                 bitwidth=8,
                 work_dir=os.path.join(os.getcwd(), 'work'),
                 quant_iter=1):
        #
        super(XGraphDefaultQuantizer, self).__init__(xgraph)

        # Setup executable graph
        self.runtime = pyxir.build(self.xgraph, target='cpu')

        self.inputs_func = inputs_func
        self.work_dir = work_dir
        self._bitwidth = bitwidth

        self._quant_param = QuantParamFactory()
        self._quant_layers = {}

        self.XFDNN_OP_2_QUANT_FUNC = {
            'Input': InputQuant(self._quant_param,
                                self._quant_layers,
                                self._bitwidth),
            'Output': DefaultQuant(self._quant_param,
                                   self._quant_layers,
                                   self._bitwidth),
            'Constant': SkipQuant(self._quant_param,
                                  self._quant_layers,
                                  self._bitwidth),

            # BASIC NN OPS
            'Dense': DefaultQuant(self._quant_param,
                                  self._quant_layers,
                                  self._bitwidth),
            'Softmax': DefaultQuant(self._quant_param,
                                    self._quant_layers,
                                    self._bitwidth),
            'ReLU': DefaultQuant(self._quant_param,
                                 self._quant_layers,
                                 self._bitwidth),
            'Tanh': DefaultQuant(self._quant_param,
                                 self._quant_layers,
                                 self._bitwidth),

            # MATH
            'Scale': ScaleQuant(self._quant_param,
                                self._quant_layers,
                                self._bitwidth),
            'Eltwise': EltwiseQuant(self._quant_param,
                                    self._quant_layers,
                                    self._bitwidth),
            'Concat': ConcatQuant(self._quant_param,
                                  self._quant_layers,
                                  self._bitwidth),
            'Mean': DefaultQuant(self._quant_param,
                                 self._quant_layers,
                                 self._bitwidth),
            'BatchNorm': BatchNormQuant(self._quant_param,
                                        self._quant_layers,
                                        self._bitwidth),

            # CONVOLUTION
            'Convolution': ConvQuant(self._quant_param,
                                     self._quant_layers,
                                     self._bitwidth),
            'Conv2DTranspose': ConvQuant(self._quant_param,
                                         self._quant_layers,
                                         self._bitwidth),
            'Pooling': PoolQuant(self._quant_param,
                                 self._quant_layers,
                                 self._bitwidth),

            # OTHER
            'Reshape': DefaultQuant(self._quant_param,
                                    self._quant_layers,
                                    self._bitwidth),
            'Squeeze': DefaultQuant(self._quant_param,
                                    self._quant_layers,
                                    self._bitwidth),
            'Flatten': DefaultQuant(self._quant_param,
                                    self._quant_layers,
                                    self._bitwidth),
            'Transpose': DefaultQuant(self._quant_param,
                                      self._quant_layers,
                                      self._bitwidth),
        }

    def quantize(self, stop=None, subgraphs_only=True):
        # type: (str, boolean) -> None
        """
        Start quantization of the executable graph model

        Arguments
        ---------
        stop: str (optional, default = None)
            the name of the operation at which to stop quantization
        """

        self._quantize(stop, subgraphs_only)

        # quant_files = {}
        q_output = QuantizerOutput(self.xgraph.get_name())
        for qkey in self._quant_layers.keys():
            if qkey != 'None':
                quant_file = os.path.join(self.work_dir, qkey + '_quant.json')
                self._quant_param.save_to_dpu_v1_json(self._quant_layers[qkey],
                                                      quant_file)
                q_output.add(qkey, quant_file, None, None)

        self.xgraph.set_quantizer_output(q_output)

        logger.info("QUANTIZATION DONE")

        return self.xgraph

    def _quantize(self, stop=None, subgraphs_only=True):
        # type: (str, boolean) -> None
        """
        Start quantization of the executable graph model

        Arguments
        ---------
        stop: str (optional, default = None)
            the name of the operation at which to stop quantization
        """

        # TODO One iter for now
        inpts = self.inputs_func(0)

        logger.info("Running network")
        for idx, layer, inpts, outpt, quant_output \
                in self.runtime.run_stepwise(inpts, stop=stop):

            op_name, op_type = layer.name, layer.type
            logger.info("-----------------------")
            logger.info("Operation idx: {}, name: {}, type: {}"
                        .format(idx, op_name, op_type))
            logger.info("Operation inputs shape: {}, output shape: {}"
                        .format([inpt.shape for inpt in inpts], outpt.shape))

            # If subgraps_only == True, we only quantize the subgraphs
            if subgraphs_only:
                qkey = layer.subgraph if layer.subgraph is not None else 'None'
            else:
                qkey = self.xgraph.get_name()

            if qkey not in self._quant_layers:
                self._quant_layers[qkey] = []

            # TODO we are doing quantization on all layers even if we don't
            #   have to do it (subgraph, ...)
            # Run quantization function
            self.XFDNN_OP_2_QUANT_FUNC[op_type](
                layer, inpts, quant_output, qkey)

        logger.info("-----------------------")
        logger.info("Done running network")
