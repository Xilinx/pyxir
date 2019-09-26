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
Module for decorating an xgraph for quantization simulation


"""

from __future__ import absolute_import

import os
import numpy as np
import logging

from pyxir.shared import fancy_logging

from pyxir.shared import QuantParams
from pyxir.graph.layer.xlayer import XLayer, defaultXLayer
from pyxir.graph.passing.base_pass import XGraphBasePass

from .quant_sim_transform_registry import QUANTIZE_LAYER,\
    register_quant_sim_transform

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")

# CONCAT


@register_quant_sim_transform("Concat")
def get_concat_quantization_layers(bottom_Ps, P, top_Ps, quant_params):
    # type: (Dict[str, XLayer], XLayer, Dict[str, XLayer], Dict[str,dict])
    #   -> List[XLayer]
    new_Ps = []

    if P.name in quant_params:
        # Quantize bottoms
        for idx, bottom in enumerate(P.bottoms):
            attrs = {
                # 'quant_params': {
                #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
                #     'quant_threshold': [quant_params[P.name]['th_layer_in']]
                # },
                'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
                'quant_threshold': [quant_params[P.name]['th_layer_in']],
                'dtype': 'int8',
                'input_types': ['float32'],
                'axis': 1  # TODO: NCHW
            }
            quant_bottom = defaultXLayer()
            quant_bottom = quant_bottom._replace(
                type=['Quantize'],
                name=bottom + "_quantize",
                shapes=bottom_Ps[idx].shapes,
                attrs=attrs,
                bottoms=[bottom],
                tops=[],
                targets=[]
            )
            new_Ps.append(quant_bottom)

        P = P._replace(
            bottoms=[bottom + "_quantize" for bottom in P.bottoms],
            tops=[]
        )
    else:
        P = P._replace(
            tops=[]
        )

    new_Ps.append(P)

    if P.name in quant_params:
        # NO QuantizeInter layer
        # TODO How to handle quantization for concat layers after concat
        #   layers?
        #   See DenseNet kind of architectures.

        # UNQUANTIZE LAYER
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[P.name]['th_layer_out']]
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': [quant_params[P.name]['th_layer_out']],
            'dtype': 'float32',
            'input_types': ['int8'],
            'axis': 1  # TODO: NCHW
        }
        unquant_P = defaultXLayer()
        unquant_P = unquant_P._replace(
            type=['UnQuantize'],
            name=P.name + "_unquantize",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name],
            tops=[],
            targets=[]
        )
        new_Ps.append(unquant_P)

    return new_Ps


@register_quant_sim_transform("Scale")
def get_scale_quantization_layers(bottom_Ps, P, top_Ps, quant_params):
    # type: (Dict[str, XLayer], XLayer, Dict[str, XLayer], Dict[str,dict])
    #   -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Ps = []

    G, B = P.data.gamma, P.data.beta
    gamma_name, beta_name = P.name + "_gamma", P.name + "_beta"

    # Scaling is executed as an elementwise layer in  combination with
    #   quantization scaling
    #  ! Ignore gamma scaling values (they are already incorporated in
    #   quantization parameters)

    new_Ps = []

    # BETA
    b_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    b_in_P = defaultXLayer()
    b_in_P = b_in_P._replace(
        type=['Constant'],
        name=beta_name,
        shapes=list(B.shape),
        bottoms=[],
        tops=[],
        attrs=b_in_attrs,
        targets=[],
        layer=[],
        data=[B]
    )
    new_Ps.append(b_in_P)

    if P.name in quant_params:
        b_quant_attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     # 'quant_threshold': [quant_params[P.name]['th_layer_in']],
            #     # 'th_params': quant_params[P.name]['th_params']
            #     'th_out': [quant_params[P.name]['th_layer_out']],
            #     'scale': [quant_params[P.name]['scale']],
            #     'postscale_shift': [quant_params[P.name]['postscale_shift']]
            #     # [th_param * 127 for th_param in quant_params[P.name]
            #     #   ['th_params']]
            #     # TODO Add wuant beta layer to avoid multiplication by 127
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_th_out': [quant_params[P.name]['th_layer_out']],
            'quant_scale': quant_params[P.name]['scale'],
            'quant_postscale_shift': quant_params[P.name]['postscale_shift'],
            'dtype': 'int32',
            'input_types': ['float32'],
            'axis': 0  # TODO: C
        }
        b_quant_P = defaultXLayer()
        b_quant_P = b_quant_P._replace(
            type=['QuantizeScaleBias'],
            name=beta_name + "_quantize",
            shapes=list(B.shape),
            attrs=b_quant_attrs,
            bottoms=[beta_name],
            tops=[],
            targets=[]
        )
        new_Ps.append(b_quant_P)

    if P.name in quant_params:
        # INPUT

        # Quantize bottoms
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[P.name]['th_layer_in']]
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': [quant_params[P.name]['th_layer_in']],
            'dtype': 'int8',
            'input_types': ['float32'],
            'axis': 0  # TODO: NCHW
        }
        assert(len(P.bottoms) == 1)
        quant_bottom = defaultXLayer()
        quant_bottom = quant_bottom._replace(
            type=['Quantize'],
            name=P.bottoms[0] + "_quantize",
            shapes=bottom_Ps[0].shapes,
            attrs=attrs,
            bottoms=[P.bottoms[0]],
            tops=[],
            targets=[]
        )
        new_Ps.append(quant_bottom)

        P = P._replace(
            bottoms=[bottom + "_quantize" for bottom in P.bottoms] +
            [beta_name + '_quantize'],
            tops=[]
        )
    else:
        P = P._replace(
            bottoms=P.bottoms + [beta_name],
            tops=[]
        )

    # Move relu from eltwise layer to QuantizeInter layer if applicable
    #   because of negative scaling case
    is_relu = 'activation' in P.attrs and P.attrs['activation'] == 'ReLU'

    # P = P._replace(
    #     type = ['Eltwise'],
    #     data = P.data.beta,
    #     targets = [],
    #     layer = []
    # )
    P.attrs['dtype'] = 'int32'
    P = P._replace(
        type=['BiasAdd'],
        data=[P.data.beta],
        targets=[],
        layer=[]
    )
    new_Ps.append(P)

    if P.name in quant_params:
        # Quantize inter layers
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'scale': quant_params[P.name]['scale'],
            #     'postscale_shift': quant_params[P.name]['postscale_shift'],
            #     'prescale_shift': quant_params[P.name]['prescale_shift']
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_scale': quant_params[P.name]['scale'],
            'quant_postscale_shift': quant_params[P.name]['postscale_shift'],
            'quant_prescale_shift': quant_params[P.name]['prescale_shift'],
            'dtype': 'int8',  # TODO??
            'input_types': ['int32'],
            'axis': 1  # TODO NCHW
        }

        if is_relu:
            attrs['activation'] = 'ReLU'

        quant_inter_P = defaultXLayer()
        quant_inter_P = quant_inter_P._replace(
            type=['QuantizeInter'],
            name=P.name + "_quantize_inter",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name],
            tops=[]
        )
        new_Ps.append(quant_inter_P)

        # UNQUANTIZE LAYER
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[P.name]['th_layer_out']]
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': [quant_params[P.name]['th_layer_out']],
            'dtype': 'float32',
            'input_types': ['int8'],
            'axis': 0  # TODO: NCHW
        }
        unquant_P = defaultXLayer()
        unquant_P = unquant_P._replace(
            type=['UnQuantize'],
            name=P.name + "_unquantize",
            shapes=P.shapes[:],
            attrs=attrs,
            bottoms=[P.name + "_quantize_inter"]
        )
        new_Ps.append(unquant_P)

    # new_Ps.extend(get_eltwise_quantization_layers(bottom_Ps, P, top_Ps,
    #   quant_params))

    return new_Ps


@register_quant_sim_transform("Pooling")
def get_pooling_quantization_layers(bottom_Ps, P, top_Ps, quant_params):
    # type: (List[XLayer], XLayer, List[XLayer], QuantParams)
    #   -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Ps = []

    # TODO: Can we do better?
    # Maxpool layers quant params are stored in the quantization file under
    #   another name otherwise it messes up maxpool computation on FPGA
    quant_name = P.name if P.name in quant_params else P.name + "_QUANT_UTIL"
    if quant_name in quant_params:
        # Quantize bottoms
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[quant_name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[quant_name]['th_layer_in']]
            # },
            'quant_bitwidth': quant_params[quant_name]['bw_layer_in'],
            'quant_threshold': [quant_params[quant_name]['th_layer_in']],
            'dtype': 'int8',
            'input_types': ['float32'],
            'axis': 1  # TODO: NCHW
        }
        assert(len(P.bottoms) == 1)
        quant_bottom = defaultXLayer()
        quant_bottom = quant_bottom._replace(
            type=['Quantize'],
            name=P.bottoms[0] + "_quantize",
            shapes=bottom_Ps[0].shapes,
            attrs=attrs,
            bottoms=[P.bottoms[0]],
            tops=[P.name]
        )
        new_Ps.append(quant_bottom)

        # NOTE: quantization parameters include the division part of the
        #   the average pooling operation. Therefore we want to use a AvgPool
        #   layer without the division part
        #   (instead just the sum of the elements).
        P = P._replace(
            type=['PoolingNoDivision'],
            bottoms=[bottom + "_quantize" for bottom in P.bottoms],
            tops=[]
        )
    else:
        P = P._replace(
            tops=[]
        )

    new_Ps.append(P)

    if quant_name in quant_params:
        # Quantize inter layers
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[quant_name]['bw_layer_in'],
            #     'scale': quant_params[quant_name]['scale'],
            #     'postscale_shift':
            #      quant_params[quant_name]['postscale_shift'],
            #     'prescale_shift': quant_params[quant_name]['prescale_shift']
            # },
            'quant_bitwidth': quant_params[quant_name]['bw_layer_in'],
            'quant_scale': quant_params[quant_name]['scale'],
            'quant_postscale_shift':
            quant_params[quant_name]['postscale_shift'],
            'quant_prescale_shift': quant_params[quant_name]['prescale_shift'],
            'dtype': 'int8',  # TODO??
            'input_types': ['int8'],
            'axis': 1  # TODO NCHW
        }
        quant_inter_P = defaultXLayer()
        quant_inter_P = quant_inter_P._replace(
            type=['QuantizeInter'],
            name=P.name + "_quantize_inter",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name]
        )
        new_Ps.append(quant_inter_P)

        # UNQUANTIZE LAYER
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[quant_name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[quant_name]['th_layer_out']]
            # },
            'quant_bitwidth': quant_params[quant_name]['bw_layer_in'],
            'quant_threshold': [quant_params[quant_name]['th_layer_out']],
            'dtype': 'float32',
            'input_types': ['int8'],
            'axis': 1  # TODO: NCHW
        }
        unquant_P = defaultXLayer()
        unquant_P = unquant_P._replace(
            type=['UnQuantize'],
            name=P.name + "_unquantize",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name + "_quantize_inter"]
        )
        new_Ps.append(unquant_P)

    return new_Ps


@register_quant_sim_transform("Eltwise")
def get_eltwise_quantization_layers(bottom_Ps, P, top_Ps, quant_params):
    # type: (Dict[str, XLayer], XLayer,
    #   Dict[str, XLayer], Dict[str,dict])
    #   -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Ps = []

    assert(len(bottom_Ps) == 2)

    if P.name in quant_params:
        # Quantize bottoms
        for idx, bottom in enumerate(P.bottoms):
            attrs = {
                # 'quant_params': {
                #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
                #     'quant_threshold': [quant_params[P.name]['th_layer_in']]
                # },
                'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
                'quant_threshold': [quant_params[P.name]['th_layer_in']],
                'dtype': 'int8',
                'input_types': ['float32'],
                'axis': 1  # NCHW
                # TODO: should elemwise be broadcastable??
            }
            quant_bottom = defaultXLayer()
            quant_bottom = quant_bottom._replace(
                type=['Quantize'],
                name=bottom + "_quantize",
                shapes=bottom_Ps[idx].shapes,
                attrs=attrs,
                bottoms=[bottom]
            )
            new_Ps.append(quant_bottom)

        P = P._replace(
            bottoms=[bottom + "_quantize" for bottom in P.bottoms],
            tops=[]
        )
    else:
        P = P._replace(
            tops=[]
        )

    new_Ps.append(P)

    if P.name in quant_params:
        # Quantize inter layers
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'scale': quant_params[P.name]['scale'],
            #     'postscale_shift': quant_params[P.name]['postscale_shift'],
            #     'prescale_shift': quant_params[P.name]['prescale_shift']
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_scale': quant_params[P.name]['scale'],
            'quant_postscale_shift': quant_params[P.name]['postscale_shift'],
            'quant_prescale_shift': quant_params[P.name]['prescale_shift'],
            'dtype': 'int8',  # TODO??
            'input_types': ['int32'],
            'axis': 1  # TODO NCHW
        }
        quant_inter_P = defaultXLayer()
        quant_inter_P = quant_inter_P._replace(
            type=['QuantizeInter'],
            name=P.name + "_quantize_inter",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name]
        )
        new_Ps.append(quant_inter_P)

        # UNQUANTIZE LAYER
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[P.name]['th_layer_out']]
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': [quant_params[P.name]['th_layer_out']],
            'dtype': 'float32',
            'input_types': ['int8'],
            'axis': 1  # TODO: NCHW
        }
        unquant_P = defaultXLayer()
        unquant_P = unquant_P._replace(
            type=['UnQuantize'],
            name=P.name + "_unquantize",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name + "_quantize_inter"]
        )
        new_Ps.append(unquant_P)

    return new_Ps


@register_quant_sim_transform("Convolution")
def get_convolution_quantization_layers(bottom_Ps, P, top_Ps, quant_params):
    # type: (Dict[str, XLayer], XLayer,
    #   Dict[str, XLayer], Dict[str,dict]) -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Ps = []

    W, B = P.data.weights, P.data.biases
    kernel_name = P.name + "_kernel"
    bias_name = P.name + "_biases"

    # KERNEL
    k_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    k_in_P = defaultXLayer()
    k_in_P = k_in_P._replace(
        type=['Input'],
        name=kernel_name,
        shapes=list(W.shape)
    )
    new_Ps.append(k_in_P)

    if P.name in quant_params:
        k_quant_attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': quant_params[P.name]['th_params']
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': quant_params[P.name]['th_params'],
            'dtype': 'int8',
            'input_types': ['float32'],
            'axis': 0  # TODO: OIHW
        }
        k_quant_P = defaultXLayer()
        k_quant_P = k_quant_P._replace(
            type=['Quantize'],
            name=kernel_name + "_quantize",
            shapes=list(W.shape),
            attrs=k_quant_attrs,
            bottoms=[kernel_name]
        )
        new_Ps.append(k_quant_P)

    # BIAS
    b_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    b_in_P = defaultXLayer()
    b_in_P = b_in_P._replace(
        type=['Input'],
        name=bias_name,
        shapes=list(B.shape)
    )
    new_Ps.append(b_in_P)

    if P.name in quant_params:
        b_quant_attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': quant_params[P.name]['th_layer_in'],
            #     'th_params': quant_params[P.name]['th_params']
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': quant_params[P.name]['th_layer_in'],
            'quant_th_params': quant_params[P.name]['th_params'],
            'dtype': 'int32',
            'input_types': ['float32']
        }
        b_quant_P = defaultXLayer()
        b_quant_P = b_quant_P._replace(
            type=['QuantizeBias'],
            name=bias_name + "_quantize",
            shapes=list(B.shape),
            attrs=b_quant_attrs,
            bottoms=[bias_name]
        )
        new_Ps.append(b_quant_P)

    # CONVOLUTION
    if P.name in quant_params:
        # INPUT

        # Quantize bottoms
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[P.name]['th_layer_in']]
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': [quant_params[P.name]['th_layer_in']],
            'dtype': 'int8',
            'input_types': ['float32'],
            'axis': 1  # TODO: NCHW
        }
        assert(len(P.bottoms) == 1)
        quant_bottom = defaultXLayer()
        quant_bottom = quant_bottom._replace(
            type=['Quantize'],
            name=P.bottoms[0] + "_quantize",
            shapes=bottom_Ps[0].shapes,
            attrs=attrs,
            bottoms=[P.bottoms[0]]
        )
        new_Ps.append(quant_bottom)

        P = P._replace(
            bottoms=[bottom + "_quantize" for bottom in P.bottoms]
            + [kernel_name + '_quantize', bias_name + '_quantize'],
            tops=[]
        )
    else:
        P = P._replace(
            bottoms=P.bottoms + [kernel_name, bias_name],
            tops=[]
        )
    new_Ps.append(P)

    # QUANTIZE AFTER CONVOLUTION
    if P.name in quant_params:
        qi_attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'scale': quant_params[P.name]['scale'],
            #     'postscale_shift': quant_params[P.name]['postscale_shift'],
            #     'prescale_shift': quant_params[P.name]['prescale_shift']
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_scale': quant_params[P.name]['scale'],
            'quant_postscale_shift': quant_params[P.name]['postscale_shift'],
            'quant_prescale_shift': quant_params[P.name]['prescale_shift'],
            'dtype': 'int8',  # TODO??
            'input_types': ['float32'],
            'axis': 1  # NCHW
        }
        quant_inter_P = defaultXLayer()
        quant_inter_P = quant_inter_P._replace(
            type=['QuantizeInter'],
            # type=['QuantizeInter12MSBits'],
            name=P.name + "_quantize_inter",
            shapes=P.shapes,
            attrs=qi_attrs,
            bottoms=[P.name]
        )
        new_Ps.append(quant_inter_P)

        # UNQUANTIZE LAYER
        attrs = {
            # 'quant_params': {
            #     'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            #     'quant_threshold': [quant_params[P.name]['th_layer_out']]
            # },
            'quant_bitwidth': quant_params[P.name]['bw_layer_in'],
            'quant_threshold': [quant_params[P.name]['th_layer_out']],
            'dtype': 'float32',
            'input_types': ['int8'],
            'axis': 1  # TODO: NCHW
        }
        unquant_P = defaultXLayer()
        unquant_P = unquant_P._replace(
            type=['UnQuantize'],
            name=P.name + "_unquantize",
            shapes=P.shapes,
            attrs=attrs,
            bottoms=[P.name + "_quantize_inter"]
        )
        new_Ps.append(unquant_P)

    return new_Ps


# QUANTIZE_LAYER = {
#     'Concat': get_concat_quantization_layers,
#     'Scale': get_scale_quantization_layers,
#     'Pooling': get_pooling_quantization_layers,
#     'Eltwise': get_eltwise_quantization_layers,
#     'Convolution': get_convolution_quantization_layers
# }

class XGraphQuantSimPass(XGraphBasePass):

    """
    Responsible for transforming XGraph models so they can be executed with
    quantization simulation

    Arguments
    ---------
    name: str
        the new name of the decorated xgraph
    output_png: str
        the name of the png file for graph visualization if specified
    """

    def __init__(self,
                 fdir='./',
                 name='XGraphQuantSim',
                 output_png=None):
        super(XGraphQuantSimPass, self).__init__(
            name=name,
            output_png=output_png
        )
        self.fdir = fdir

    def execute(self, xgraph, subgraphs_only=True):
        # type: (XGraph, bool) -> XGraph
        """
        The decorator method contains the functionality to decorate the xgraph.

        # TODO: Test this decorator
        """

        # Retrieve quant params
        if subgraphs_only:
            subgraph_names = xgraph.get_subgraph_names()

            quant_params_d = {
                sg_name: QuantParams(
                    os.path.join(self.fdir, sg_name + '_quant.json')
                ) for sg_name in subgraph_names
            }
        else:
            quant_params_d = {
                xgraph.get_name(): QuantParams(
                    os.path.join(self.fdir, xgraph.get_name() + '_quant.json')
                )
            }

        def add_quant_layers(bottom_Ps, P, top_Ps):
            # type: (Dict[str, XLayer], XLayer,
            #  Dict[str, XLayer]) -> List[XLayer]
            """
            Replace the provided parameters layer with a list of parameter
            layers adding quantization layers before or after the given layer
            """
            # TODO: tops
            # nonlocal skip # python 2.7

            new_Ps = []

            # TODO: Create helper class to make ParametersLayer creation
            #   less error prone
            if P.type[0] in QUANTIZE_LAYER:
                # Check if we want to quantize subgraphs or full graph
                if subgraphs_only and P.subgraph is not None:
                    quant_params = quant_params_d[P.subgraph]
                elif not subgraphs_only:
                    quant_params = quant_params_d[xgraph.get_name()]
                else:
                    quant_params = None

                if quant_params is not None:
                    logger.info("Add quant layers for: {}".format(P.name))

                    p_layers = QUANTIZE_LAYER[P.type[0]](
                        bottom_Ps, P, top_Ps, quant_params)
                    new_Ps.extend(p_layers)

                    logger.info("Number of new layers: {}"
                                .format(len(p_layers)))
                else:
                    new_Ps.append(P)
            else:
                # P = P._replace(
                #    tops = []
                # )
                new_Ps.append(P)

            # if self.last_sim_layer is not None and self.last_sim_layer ==
            # P.name:
            # Start skipping quantization simulation from the next layer
            #    d['skip'] = True

            return new_Ps

        def merge_quant_layers(bottom_Ps, P, top_Ps):
            # type: (Dict[str, XLayer], XLayer,
            #  Dict[str, XLayer]) -> List[XLayer]
            """
            Merge the provided parameters layer with preceding or succeeding
            layer if they cancel eachother out

            TODO: Formalize ParametersLayer comparison in a separate wrapper
                  class?
            TODO: Remove quantize inter layers that are doing the identity
                  operation
            """
            new_Ps = []

            if 'UnQuantize' in P.type:
                # Check if all tops are Quantize layers that can be cancelled
                #   out
                remove = len(top_Ps) > 0
                logger.info("Layer is an UnQuantize layer, init remove = {}"
                            .format(remove))
                for i in range(len(top_Ps)):
                    t_attrs, attrs = top_Ps[i].attrs, P.attrs
                    logger.debug("-- top_P: {}, type: {}"
                                 .format(top_Ps[i].name, top_Ps[i].type))
                    logger.debug(t_attrs)
                    logger.debug(attrs)
                    # or t_attrs['quant_params']['quant_threshold'] != \
                    # attrs['quant_params']['quant_threshold']\
                    # or t_attrs['quant_params']['quant_bitwidth'] !=\
                    # attrs['quant_params']['quant_bitwidth']\
                    if 'Quantize' not in top_Ps[i].type\
                            or t_attrs['quant_threshold'] != \
                            attrs['quant_threshold']\
                            or t_attrs['quant_bitwidth'] != \
                            attrs['quant_bitwidth']\
                            or t_attrs['dtype'] != attrs['input_types'][0]\
                            or t_attrs['input_types'][0] != attrs['dtype']\
                            or t_attrs['axis'] != attrs['axis']:
                        # The succeeding Quantize layer and this
                        # UnQuantize layer DON'T cancel eachother out
                        remove = False
                        break

                if not remove:
                    logger.info("Don't remove layer")
                    new_Ps.append(P)
                else:
                    logger.info("Remove layer and top layers: {}".format(
                        [top_P.name for top_P in top_Ps]))
                    for i in range(len(top_Ps)):
                        top_Ps[i].attrs['remove_layer'] = True
            elif 'Quantize' in P.type:
                assert(len(bottom_Ps) == 1)
                logger.info("Layer is a Quantize layer")
                # If this quantize layer is indicated to be removed by
                # previous UnQuantize layer, then remove this quantize layer
                if 'remove_layer' not in P.attrs\
                        or not P.attrs['remove_layer']:
                    logger.info("Don't remove layer")
                    new_Ps.append(P)
            else:
                P = P._replace(
                    tops=[]
                )
                new_Ps.append(P)

            return new_Ps

        fancy_logger.banner("ADD QUANT LAYERS")

        output_png = self.output_png.replace(".", "_add_quant_layers.") \
            if self.output_png is not None else None
        xgraph = self._replace_layer_pass(
            xgraph=xgraph,
            replace_func=add_quant_layers,
            name=self.name + "_add_quant_layers",
            output_png=output_png
        )

        fancy_logger.banner("MERGE QUANT LAYERS")

        output_png = self.output_png.replace(".", "_merge_quant_layers.") \
            if self.output_png is not None else None
        xgraph = self._replace_layer_pass(
            xgraph=xgraph,
            replace_func=merge_quant_layers,
            name=self.name + "_merge_quant_layers",
            output_png=output_png
        )

        return xgraph
