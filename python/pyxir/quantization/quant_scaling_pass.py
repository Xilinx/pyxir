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
Module for inserting scaling layer before eltwise addition layers
(e.g. in ResNets)


"""
import logging
import numpy as np

from pyxir.graph.layer import xlayer
from pyxir.shared import fancy_logging
from pyxir.shared.quant_param_factory import QuantParamFactory

from .quant_pass import XGraphQuantPass

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


class XGraphQuantScalingPass(XGraphQuantPass):

    """
    Responsible for finishing up quantization of XGraph objects for
    correctness and accuracy.

    For example, we will handle the quantization of elementwise additions here
    because they require the insertion of a new scaling layer for best accuracy
    """

    def execute(self, xgraph):
        # type: (XGraph) -> XGraph
        """
        """

        quant_params = self.quant_params

        def _get_quant_params(name, bitwidth, channels, th_layer_in,
                              th_layer_out):
            sf_layer_in = th_layer_in / (np.power(2.0, bitwidth - 1) - 1)
            sf_layer_out = th_layer_out / (np.power(2.0, bitwidth - 1) - 1)
            sf_params = np.array([1.0]) / (np.power(2.0, bitwidth - 1) - 1)

            # Scale and postscale shift for division
            multiplier = th_layer_in / th_layer_out
            logger.debug("th_layer_in: {}, th_layer_out: {}, ratio: {}"
                         .format(th_layer_in, th_layer_out, multiplier))

            # TODO
            prescale_shift_max = 0
            scale_bitwidth = 16
            postscale_shift_max = 40

            canonical_factor = np.power(2, np.ceil(np.log2(multiplier)))
            canonical_form = multiplier / canonical_factor

            shift = np.log2(canonical_factor)
            lshift = np.clip(shift, 0, None)
            rshift = -np.clip(shift, None, 0)

            prescale_shift = np.clip(rshift, 0, prescale_shift_max)
            rshift -= prescale_shift

            postscale_shift = np.clip(rshift, 0, postscale_shift_max)
            rshift -= postscale_shift

            scale = np.clip(canonical_form * np.power(2, lshift - rshift), 0,
                            np.power(2, scale_bitwidth - 1))

            remaining_available_lshift = np.floor(np.log2(np.power(2,
                                                  scale_bitwidth - 1) / scale))
            remaining_available_rshift = postscale_shift_max - postscale_shift
            remaining_available_shift = np.fmin(remaining_available_lshift,
                                                remaining_available_rshift)

            scale *= np.power(2, remaining_available_shift)
            postscale_shift += remaining_available_shift

            prescale_shift = prescale_shift.astype(int)
            scale = np.round(scale).astype(int)
            postscale_shift = postscale_shift.astype(int)

            prescale_shift = np.clip(prescale_shift, 0, prescale_shift_max)
            scale = np.clip(scale, 0, np.power(2, scale_bitwidth) - 1)
            postscale_shift = np.clip(postscale_shift, 0, postscale_shift_max)

            logger.debug("prescale: {}, scale: {}, postscale: {}"
                         .format(prescale_shift, scale, postscale_shift))
            logger.debug("Type: prescale: {}, scale: {}, postscale: {}"
                         .format(type(prescale_shift.astype(np.int32)),
                                 type(scale.astype(np.int32)),
                                 type(postscale_shift.astype(np.int32))))

            qp = {
                "name": name,
                "bw_layer_in": bitwidth,  # unused by xfdnn
                "bw_layer_out": bitwidth,  # unused by xfdnn
                "bw_params": bitwidth,
                "th_layer_in": th_layer_in,
                "th_layer_out": th_layer_out,
                "th_params": [1] * channels,
                "sf_layer_in": sf_layer_in,  # unused by xfdnn
                "sf_layer_out": sf_layer_out,  # unused by xfdnn
                "sf_params": sf_params.tolist() * channels,
                "prescale_shift": [int(prescale_shift.astype(np.int32))]
                * channels,
                "scale": [int(scale.astype(np.int32))] * channels,
                "postscale_shift": [int(postscale_shift.astype(np.int32))]
                * channels
            }
            return qp

        # TODO: this is hacky
        top_count = {}

        def add_scaling_layer_before_eltwise_and_concat(bottom_Ps, P, top_Ps):
            # type: (List[XLayer], XLayer, List[XLayer])
            #   -> List[XLayer]
            """
            Replace the provided parameters layer with a list of XLayer
            adding scaling layers before elementwise additions
            """
            # TODO: tops
            nonlocal top_count, quant_params
            top_count[P.name] = len(top_Ps)

            new_Ps = []

            if 'Concat' in P.type or 'Eltwise' in P.type:
                logger.info("Add scaling before {} layer: {}"
                            .format(P.type[0], P.name))

                new_bottoms = []
                for bottom_P in bottom_Ps:

                    logger.debug("Bottom: {}, th_layer_out: {},"
                                 " th_layer_in: {}"
                                 .format(bottom_P.name, quant_params[
                                         bottom_P.name]['th_layer_out'],
                                         quant_params[P.name]['th_layer_in']))
                    logger.debug("Bottom tops length: {}"
                                 .format(top_count[bottom_P.name]))

                    if abs(quant_params[bottom_P.name]['th_layer_out'] -
                            quant_params[P.name]['th_layer_in']) < 0.001:
                        # if the minimum output threshold of the previous
                        #   layers is equal to the input threshold of this
                        #   Eltwise layer, we don't have to insert a scaling
                        #   layer (e.g. it's already
                        #   there)
                        # P = P._replace(
                        #     tops = [] # important
                        # )
                        # new_Ps.append(bottom_P)
                        new_bottoms.append(bottom_P.name)
                        # return new_Ps
                    elif top_count[bottom_P.name] == 1 and \
                            (bottom_P.type[0] in ['Convolution', 'Scale',
                                                  'Eltwise'] or
                             (bottom_P.type[0] in ['Pooling'] and
                              bottom_P.pool == 1)):
                        # Convolution, Scale, Eltwise and AvgPool

                        # Threshold difference and previous layer has one top
                        #   -> change the threshold of the bottom_P layer
                        # TODO
                        # raise NotImplementedError('')
                        th_layer_out = quant_params[P.name]['th_layer_in']
                        prev_th_out = quant_params[bottom_P.name][
                            'th_layer_out']

                        logger.debug("-- Adjust th_out: {} -> {}"
                                     .format(prev_th_out, th_layer_out))
                        quant_params[bottom_P.name]['th_layer_out'] = \
                            th_layer_out
                        new_bottoms.append(bottom_P.name)

                    elif top_count[bottom_P.name] >= 1:
                        # Threshold difference and previous layer has multiple
                        #   tops

                        # TODO: we are assuming a certain structure here: NCHW
                        #   we have to solve this better intrinscally
                        # TODO: if FPGA supports scaling layer with one
                        #   scaling value
                        #   we can just pass one value here = TEST
                        channels = 1

                        assert(len(bottom_P.shapes) == 4)
                        channels = bottom_P.shapes[1]  # NCHW

                        th_layer_in = quant_params[bottom_P.name][
                            'th_layer_out']
                        th_layer_out = quant_params[P.name]['th_layer_in']
                        logger.debug("-- Scaling layer: th_in {} -> th_out {}"
                                     .format(th_layer_in, th_layer_out))

                        # Add a scaling layer to scale by th_layer_out /
                        #   th_layer_in
                        b_scale_name = bottom_P.name + '_scale_for_quant'
                        b_scale = P._replace(
                            name=b_scale_name,
                            shapes=bottom_P.shapes[:],
                            sizes=bottom_P.sizes[:],
                            type=['Scale'],
                            bottoms=[bottom_P.name],
                            # bias=True,
                            data=xlayer.ScaleData(
                                gamma=np.array([1.]*channels),
                                beta=np.array([0.]*channels)
                            ),
                            attrs={'axis': 1}
                        )

                        qp = _get_quant_params(
                            b_scale_name,
                            quant_params[P.name]['bw_layer_in'],
                            channels,
                            th_layer_in,
                            th_layer_out)

                        # TODO: replace or insert
                        quant_params.insert_with_replace(b_scale_name, qp,
                                                         bottom_P.name)

                        new_Ps.append(b_scale)
                        new_bottoms.append(b_scale_name)
                    else:
                        raise ValueError("")

                P = P._replace(
                    bottoms=new_bottoms,
                    tops=[]
                )
                new_Ps.append(P)

            else:
                P = P._replace(
                    # important, tops wil be recomputed later on
                    tops=[]
                )
                new_Ps.append(P)

            return new_Ps

        fancy_logger.banner("GRAPH PASS ADD SCALING LAYERS")

        xgraph = self._replace_layer_pass(
            xgraph,
            add_scaling_layer_before_eltwise_and_concat,
            # name=self.name + "_scale_before_eltwise_and_concat",
            name=xgraph.get_name(),
            # Add blobs to graph for compilation afterwards
            blobs=False,
            output_png=self.output_png
        )

        # Rebuild and save the adjusted quantization parameters
        qp_factory = QuantParamFactory()
        qp_factory.rebuild_from_scratch(xgraph, quant_params, self.quantizecfg)

        return xgraph
