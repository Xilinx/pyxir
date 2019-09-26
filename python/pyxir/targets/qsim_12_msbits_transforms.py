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
Override quantization simulation cpu execution
with clipping of 12 most significant bits
"""

from pyxir.quantization.simulation.quant_sim_transform_registry import\
    register_quant_sim_transform

from pyxir.graph.layer.xlayer import XLayer, defaultXLayer


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
            # type=['QuantizeInter'],
            type=['QuantizeInter12MSBits'],
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
