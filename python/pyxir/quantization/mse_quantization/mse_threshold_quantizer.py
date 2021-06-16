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
Module for quantization with mean squared error (MSE) threshold clipping


"""

import os
import pyxir
import logging
import numpy as np
import tensorflow as tf

from pyxir.shared import fancy_logging
from pyxir.quantization.base_quantizer import XGraphBaseQuantizer
from pyxir.shared import QuantParams, QuantParamFactory, LayerParams
from pyxir.shared.quantizer_output import QuantizerOutput
from pyxir.quantization.quant_scaling_pass import XGraphQuantScalingPass
import pyxir.contrib.tools.classification as xfdnn_classification

from .xgraph_pass_add_mse_quant_layers import XGraphPassAddMSEQuantLayers
# from .pyxir_pass_add_eltwise_scale_layers import#
#   XGraphPassAddEltwiseScaleLayers

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


class XGraphMSEThresholdQuantizer(XGraphBaseQuantizer):

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
    mse_opt_num: int
        the number of trials for optimizing mean squared (MSE) error between
        full precision and quantized outputs
    """

    def __init__(self,
                 xgraph,
                 inputs_func,
                 bitwidth=8,
                 work_dir=os.path.join(os.getcwd(), 'work'),
                 quant_iter=1,
                 mse_opt_num=50):
        super(XGraphMSEThresholdQuantizer, self).__init__(xgraph)

        self.inputs_func = inputs_func
        self.work_dir = work_dir
        self.bitwidth = bitwidth
        self.mse_opt_num = mse_opt_num

        self.quant_xgraph = None
        self.runtime = None

        self._quant_param = QuantParamFactory()
        self._quant_layers = {}

        self.q_output = QuantizerOutput(name=xgraph.get_name())

    def quantize(self, stop=None, subgraphs_only=True):
        # (str, boolean) -> None
        """ Start MSE quantization """

        self._quantize(self.xgraph, stop, subgraphs_only=subgraphs_only)

        for qkey in self._quant_layers.keys():

            if qkey != 'None':
                quant_file = os.path.join(self.work_dir, qkey + '_quant.json')
                self._quant_param.save_to_dpu_v1_json(self._quant_layers[qkey],
                                                      quant_file)

                self.q_output.add(qkey, orig_pb=None, q_eval=quant_file)

                # TODO Add scaling layers
                # TODO Move adding scaling layer to before optimization
                fancy_logger.banner("ADD QUANTIZATION SCALING LAYERS FOR: {}"
                                    .format(qkey))

                quant_params = QuantParams(quant_file)
                graph_pass = XGraphQuantScalingPass(
                    quant_params,
                    quant_file,
                    output_png='tvm_quant_eltwise_scaling.png'
                    if logger.getEffectiveLevel() <= 10 else None)
                xgraph = graph_pass.execute(self.xgraph)

                self.xgraph = xgraph

        self.xgraph.set_quantizer_output(self.q_output)

        fancy_logger.banner("FINISHED QUANTIZATION")

        return xgraph

    def _quantize(self, xgraph, stop=None, subgraphs_only=True):
        # (str, boolean) -> None
        """ Start MSE quantization """

        # Graph pass to construct new graph with quantization layers
        graph_pass = XGraphPassAddMSEQuantLayers(
            bitwidth=self.bitwidth,
            mse_opt_num=self.mse_opt_num,
            subgraphs_only=subgraphs_only,
            output_png='tvm_mse_quant.png'
            if logger.getEffectiveLevel() <= 10 else None,
            name=xgraph.get_name()
        )
        xgraph = graph_pass.execute(xgraph=xgraph)

        self.quant_xgraph = xgraph
        self.runtime = pyxir.build(self.quant_xgraph, target='cpu')

        # Run graph to set Variable layer thresholds in graph
        fancy_logger.banner("EXECUTE QUANTIZATION GRAPH")

        inpts = self.inputs_func(0)
        out, params = self.runtime.optimize(inpts)

        logger.info("Done executing graph")
        # logger.info(out.shape, out)
        # logger.info(thresholds)

        logger.info("Retrieving quantization parameters...")

        self._retrieve_quant_params(params, xgraph, subgraphs_only)

    def _retrieve_quant_params(self, thresholds, xgraph, subgraphs_only):
        # type: (dict, XGraph) -> None
        """ """
        # TODO
        logger.debug("Thresholds: {}".format(thresholds))

        # TODO implement as a graph pass??
        for X in xgraph.get_layers():

            bottom_Xs = xgraph.get_bottom_layers(X.name)
            top_Xs = xgraph.get_top_layers(X.name)

            if subgraphs_only and X.subgraph is not None:
                qkey = X.subgraph
            elif subgraphs_only:
                qkey = "None"
            else:
                qkey = xgraph.get_name()

            if qkey not in self._quant_layers:
                self._quant_layers[qkey] = []

            # if 'Input' in X.type and len(top_Xs) == 1 and\
            #         'MSEQuantize' in top_Xs[0].type:

            #     self._quant_layers[qkey].append((X.name, 'Input', None))

            #     assert(len(top_Xs[0].bottoms) == 2)
            #     th_out = thresholds[top_Xs[0].bottoms[1]]

            #     self._quant_param.bw_layer_in[X.name] = self.bitwidth
            #     self._quant_param.th_layer_in[X.name] = th_out
            #     self._quant_param.bw_layer_out[X.name] = self.bitwidth
            #     self._quant_param.th_layer_out[X.name] = th_out

            if 'Convolution' in X.type and len(top_Xs) == 1 and\
                    'MSEQuantize' in top_Xs[0].type:

                self._quant_layers[qkey].append((X.name, 'Convolution', None))
                assert len(bottom_Xs) == 3
                assert 'MSEQuantize' in bottom_Xs[0].type or\
                       'MSEMockQuantize' in bottom_Xs[0].type
                assert 'MSEQuantize' in bottom_Xs[1].type

                assert(len(top_Xs[0].bottoms) == 2)

                th_in = thresholds[bottom_Xs[0].bottoms[1]]
                th_params = thresholds[bottom_Xs[1].bottoms[1]]
                th_out = thresholds[top_Xs[0].bottoms[1]]

                self._quant_param.bw_layer_in[X.name] = self.bitwidth
                self._quant_param.th_layer_in[X.name] = th_in
                self._quant_param.bw_params[X.name] = self.bitwidth
                self._quant_param.th_params[X.name] = th_params
                self._quant_param.bw_layer_out[X.name] = self.bitwidth
                self._quant_param.th_layer_out[X.name] = th_out

            elif 'Scale' in X.type and len(top_Xs) == 1 and\
                    'MSEQuantize' in top_Xs[0].type:

                gamma = X.data.gamma
                self._quant_layers[qkey].append((X.name, 'Scale',
                                                 [LayerParams(gamma)]))

                assert(len(bottom_Xs) == 3)
                assert 'MSEQuantize' in bottom_Xs[0].type or\
                       'MSEMockQuantize' in bottom_Xs[0].type
                assert('Input' in bottom_Xs[1].type)
                assert('MSEQuantizeBias' in bottom_Xs[2].type)
                assert(len(top_Xs[0].bottoms) == 2)

                th_in = thresholds[bottom_Xs[0].bottoms[1]]
                th_params = X.data.gamma
                th_out = thresholds[top_Xs[0].bottoms[1]]

                self._quant_param.bw_layer_in[X.name] = self.bitwidth
                self._quant_param.th_layer_in[X.name] = th_in
                self._quant_param.bw_params[X.name] = self.bitwidth
                self._quant_param.th_params[X.name] = th_params
                self._quant_param.bw_layer_out[X.name] = self.bitwidth
                self._quant_param.th_layer_out[X.name] = th_out

            elif 'MSEQuantizeEltwise' in X.type and len(top_Xs) == 1 and\
                    'MSEQuantize' in top_Xs[0].type:
                # 'MSEQuantizeEltwise'

                self._quant_layers[qkey].append((X.name, 'Eltwise', None))

                assert(len(bottom_Xs) == 5)
                assert 'MSEQuantize' in bottom_Xs[1].type or\
                       'MSEMockQuantize' in bottom_Xs[1].type
                assert 'MSEQuantize' in bottom_Xs[3].type or\
                       'MSEMockQuantize' in bottom_Xs[3].type

                # th_in_1 = thresholds[bottom_Xs[0].bottoms[1]]
                # th_in_2 = thresholds[bottom_Xs[1].bottoms[1]]
                # th_in = np.maximum(th_in_1, th_in_2)
                th_in = thresholds[X.bottoms[4]]
                th_out = thresholds[top_Xs[0].bottoms[1]]
                assert(len(top_Xs[0].bottoms) in [2, 4])

                self._quant_param.bw_layer_in[X.name] = self.bitwidth
                self._quant_param.th_layer_in[X.name] = th_in
                self._quant_param.bw_layer_out[X.name] = self.bitwidth
                self._quant_param.th_layer_out[X.name] = th_out

            elif 'Concat' in X.type and len(top_Xs) == 1 and \
                    'MSEQuantize' in top_Xs[0].type:

                self._quant_layers[qkey].append((X.name, 'Concat', None))
                logger.debug("CONCAT!!")

                for bottom_X in bottom_Xs:
                    assert 'MSEQuantize' in bottom_X.type or\
                           'MSEMockQuantize' in bottom_X.type

                th_in = thresholds[top_Xs[0].bottoms[1]]
                th_out = thresholds[top_Xs[0].bottoms[1]]
                assert(len(top_Xs[0].bottoms) in [2, 4])

                self._quant_param.bw_layer_in[X.name] = self.bitwidth
                self._quant_param.th_layer_in[X.name] = th_in
                self._quant_param.bw_layer_out[X.name] = self.bitwidth
                self._quant_param.th_layer_out[X.name] = th_out

            elif 'Pooling' in X.type and len(top_Xs) == 1 and \
                    ('MSEMockQuantize' in top_Xs[0].type or
                     'MSEQuantize' in bottom_Xs[0].type):

                assert(len(top_Xs) == 1)
                assert(len(bottom_Xs) == 1)
                assert(len(top_Xs[0].bottoms) == 2)
                # assert('MSEQuantize' in bottom_Xs[0].type or\
                #     'MSEMockQuantize' in bottom_Xs[0].type)

                if X.attrs['pool_type'] == 'Max':
                    # Maxpool
                    pool_divisor = [1]
                elif X.attrs['pool_type'] == 'Avg':
                    # Avg pool
                    pool_divisor = [np.prod(X.attrs['kernel_size'])]

                self._quant_layers[qkey].append((X.name, 'Pooling',
                                                 LayerParams(pool_divisor)))

                th_in = thresholds[bottom_Xs[0].bottoms[1]]
                th_out = thresholds[top_Xs[0].bottoms[1]]

                if X.attrs['pool_type'] == 'Max':
                    assert(th_in == th_out)

                self._quant_param.bw_layer_in[X.name] = self.bitwidth
                self._quant_param.th_layer_in[X.name] = th_in
                self._quant_param.bw_layer_out[X.name] = self.bitwidth
                self._quant_param.th_layer_out[X.name] = th_out

        # qp_factory.save_to_dpu_v1_json(quant_layers, quant_file)
