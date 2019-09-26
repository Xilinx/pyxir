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
Module for inserting MSE threshold optimization layers into graph


"""

import numpy as np

import logging

from pyxir.shared import fancy_logging

from pyxir.graph.layer.xlayer import XLayer, defaultXLayer
from pyxir.graph.passing.base_pass import XGraphBasePass


logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")

# CONCAT


def get_concat_quantize_layers(bottom_Xs, X, top_Xs, mse_opt_num,
                               bitwidth, new_xlayer_names):
    # type: (List[XLayer], XLayer, List[XLayer], int, int dict) -> List[XLayer]

    assert bitwidth == 8
    new_Xs = []

    X = X._replace(
        bottoms=[new_xlayer_names[bottom] for bottom in X.bottoms]
    )
    new_Xs.append(X)

    # Threshold layer
    th_out_var_name = X.name + '_th_out'
    th_out_var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    th_out_var_layer = defaultXLayer()
    th_out_var_layer = th_out_var_layer._replace(
        type=['Variable'],
        name=th_out_var_name,
        shapes=[1],
        attrs=th_out_var_attrs,
        bottoms=[],
        data=[np.array(1.)],
        tops=[]
    )
    new_Xs.append(th_out_var_layer)

    # Quantize layer
    quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO??
        'axis': 0,  # TODO NCHW
        'mse_opt_num': mse_opt_num
    }
    quant_X = defaultXLayer()
    quant_X = quant_X._replace(
        type=['MSEQuantize'],
        name=X.name + "_quantize",
        shapes=X.shapes[:],
        attrs=quant_attrs,
        bottoms=[X.name, th_out_var_name],
        tops=[]
    )
    new_Xs.append(quant_X)

    return new_Xs


def get_scale_quantize_layers(bottom_Xs, X, top_Xs, mse_opt_num,
                              bitwidth, new_xlayer_names):
    # type: (List[XLayer], XLayer, List[XLayer], int, int dict) -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Xs = []

    # TODO train beta if threshold scaling layer
    G, B = X.data.gamma, X.data.beta
    gamma_name, beta_name = X.name + "_gamma", X.name + "_beta"

    # Scaling is executed as an elementwise layer in  combination with
    #   quantization scaling
    #  ! Ignore gamma scaling values (they are already incorporated in
    #       quantization parameters)

    # INPUT

    th_in_var_name = X.name + '_th_in'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_in_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        data=[np.array(1.)],
        tops=[]
    )
    new_Xs.append(var_layer)

    quant_in_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO: input types?
        'axis': 0,  # TODO: NCHW
        'mse_opt_num': mse_opt_num
    }
    quant_in_layer = defaultXLayer()
    quant_in_layer = quant_in_layer._replace(
        type=['MSEQuantize'],
        name=X.name + '_quantize_in',
        shapes=bottom_Xs[0].shapes,
        attrs=quant_in_attrs,
        bottoms=[new_xlayer_names[X.bottoms[0]], th_in_var_name],
        tops=[]
    )
    new_Xs.append(quant_in_layer)

    # GAMMA
    g_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    g_in_X = defaultXLayer()
    g_in_X = g_in_X._replace(
        type=['Input'],
        name=gamma_name,
        shapes=list(G.shape),
        bottoms=[],
        tops=[],
        attrs=g_in_attrs
    )
    new_Xs.append(g_in_X)

    # BETA
    b_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    b_in_X = defaultXLayer()
    b_in_X = b_in_X._replace(
        type=['Input'],
        name=beta_name,
        shapes=list(B.shape),
        bottoms=[],
        tops=[],
        attrs=b_in_attrs
    )
    new_Xs.append(b_in_X)

    b_quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO: input types?
        'axis': 0  # TODO: NCHW
    }
    b_quant_X = defaultXLayer()
    b_quant_X = b_quant_X._replace(
        type=['MSEQuantizeBias'],
        name=beta_name + '_quantize',
        shapes=list(B.shape),
        attrs=b_quant_attrs,
        bottoms=[beta_name, th_in_var_name, gamma_name],
        tops=[]
    )
    new_Xs.append(b_quant_X)

    X = X._replace(
        bottoms=[X.name + '_quantize_in'] +
        [gamma_name, beta_name + '_quantize']
    )
    new_Xs.append(X)

    # Threshold layer
    th_out_var_name = X.name + '_th_out'
    th_out_var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    th_out_var_layer = defaultXLayer()
    th_out_var_layer = th_out_var_layer._replace(
        type=['Variable'],
        name=th_out_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        data=[np.array(1.)],
        tops=[]
    )
    new_Xs.append(th_out_var_layer)

    # Quantize layer
    quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO??
        'axis': 0,  # TODO NCHW
        'mse_opt_num': mse_opt_num
    }
    quant_X = defaultXLayer()
    quant_X = quant_X._replace(
        type=['MSEQuantize'],
        name=X.name + "_quantize",
        shapes=X.shapes[:],
        attrs=quant_attrs,
        bottoms=[X.name, th_out_var_name],
        tops=[]
    )
    new_Xs.append(quant_X)

    return new_Xs


def get_pooling_quantize_layers(bottom_Xs, X, top_Xs, mse_opt_num, bitwidth,
                                new_xlayer_names):
    # type: (Dict[str, XLayer], XLayer, Dict[str, XLayer])
    #   -> List[XLayer]
    """
    TODO: Make more modular
    """

    new_Xs = []

    assert(len(bottom_Xs) == 1)

    # Input
    th_in_var_name = X.name + '_th_in'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_in_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        tops=[],
        data=[np.array(1.)],
        subgraph=X.subgraph
    )
    new_Xs.append(var_layer)

    quant_in_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',
        'axis': 1 if X.attrs['data_layout'] == 'NCHW' else 3,
        'mse_opt_num': mse_opt_num
    }
    quant_in_layer = defaultXLayer()
    quant_in_layer = quant_in_layer._replace(
        type=['MSEQuantize'],
        name=X.name + '_quantize_in',
        shapes=bottom_Xs[0].shapes[:],
        attrs=quant_in_attrs,
        bottoms=[new_xlayer_names[X.bottoms[0]], th_in_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(quant_in_layer)

    # Pooling layer

    X = X._replace(
        bottoms=[X.name + '_quantize_in']
    )
    new_Xs.append(X)

    # Threshold layer
    th_out_var_name = X.name + '_th_out'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_out_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        data=[np.array(1.)],
        tops=[]
    )
    # variable_layers.add(th_in_var_name)
    new_Xs.append(var_layer)

    # Quantize layer
    # Mock quantization for max pooling layers because for max pooling
    #   input and output threshold should be equal
    quant_type = 'MSEMockQuantize' if X.attrs['pool_type'] == 'Max' \
        else 'MSEQuantize'
    quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO??
        'axis': 1 if X.attrs['data_layout'] == 'NCHW' else 3,
        'mse_opt_num': mse_opt_num
    }
    quant_X = defaultXLayer()
    quant_X = quant_X._replace(
        type=[quant_type],
        name=X.name + "_quantize",
        shapes=X.shapes,
        attrs=quant_attrs,
        bottoms=[X.name, th_out_var_name],
        tops=[]
    )
    new_Xs.append(quant_X)

    return new_Xs


def get_eltwise_quantize_layers(bottom_Xs, X, top_Xs, mse_opt_num, bitwidth,
                                new_xlayer_names):
    # type: (List[XLayer], XLayer, List[XLayer], int, int dict) -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Xs = []

    assert(len(bottom_Xs) == 2)

    # Input 0
    th_in_var_name = X.name + '_th_in_0'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_in_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        tops=[],
        data=[np.array(1.)],
        subgraph=X.subgraph
    )
    new_Xs.append(var_layer)

    quant_in_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',
        'axis': 0,
        'mse_opt_num': mse_opt_num
    }
    quant_in_layer = defaultXLayer()
    quant_in_layer = quant_in_layer._replace(
        type=['MSEQuantize'],
        name=X.name + '_quantize_in_0',
        shapes=bottom_Xs[0].shapes[:],
        attrs=quant_in_attrs,
        bottoms=[new_xlayer_names[X.bottoms[0]], th_in_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(quant_in_layer)

    # Input 1
    th_in_var_name = X.name + '_th_in_1'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_in_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        tops=[],
        data=[np.array(1.)],
        subgraph=X.subgraph
    )
    new_Xs.append(var_layer)

    quant_in_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',
        'axis': 0,
        'mse_opt_num': mse_opt_num
    }
    quant_in_layer = defaultXLayer()
    quant_in_layer = quant_in_layer._replace(
        type=['MSEQuantize'],
        name=X.name + '_quantize_in_1',
        shapes=bottom_Xs[1].shapes[:],
        attrs=quant_in_attrs,
        bottoms=[new_xlayer_names[X.bottoms[1]], th_in_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(quant_in_layer)

    # Threshold in layer
    th_in_var_name = X.name + '_th_in'
    th_in_var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    th_in_var_layer = defaultXLayer()
    th_in_var_layer = th_in_var_layer._replace(
        type=['Variable'],
        name=th_in_var_name,
        shapes=[1],
        attrs=th_in_var_attrs,
        bottoms=[],
        data=[np.array(1.)],
        tops=[]
    )
    new_Xs.append(th_in_var_layer)

    quant_eltwise_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO??
        'axis': 1,  # TODO NCHW
        'mse_opt_num': mse_opt_num
    }

    X = X._replace(
        type=['MSEQuantizeEltwise'],
        bottoms=[X.name + "_quantize_in_0", X.name + "_quantize_in_0"]
        + [X.name + "_quantize_in_1",  X.name + "_quantize_in_1"]
        + [th_in_var_name],
        # bottoms = [X.bottoms[0], X.name + "_quantize_in_0"] \
        #     + [X.bottoms[1],  X.name + "_quantize_in_1"] \
        #     + [th_in_var_name],
        attrs=quant_eltwise_attrs
    )
    new_Xs.append(X)

    # Threshold layer
    th_out_var_name = X.name + '_th_out'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_out_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        data=[np.array(1.)],
        tops=[]
    )
    new_Xs.append(var_layer)

    # Quantize layer
    quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO??
        'axis': 1,  # TODO NCHW
        'mse_opt_num': mse_opt_num
    }
    quant_X = defaultXLayer()
    quant_X = quant_X._replace(
        type=['MSEQuantize'],
        name=X.name + "_quantize",
        shapes=X.shapes,
        attrs=quant_attrs,
        bottoms=[X.name, th_out_var_name],  # eltwise_scale_name, beta_var_name
        tops=[]
    )
    new_Xs.append(quant_X)

    return new_Xs


def get_convolution_quantize_layers(bottom_Xs, X, top_Xs, mse_opt_num,
                                    bitwidth, new_xlayer_names):
    # type: (List[XLayer], XLayer, List[XLayer], int, int, dict)
    #   -> List[XLayer]
    """
    TODO: Make more modular
    """
    new_Xs = []

    W, B = X.data.weights, X.data.biases
    kernel_name = X.name + "_kernel"
    bias_name = X.name + "_biases"

    # INPUT

    th_in_var_name = X.name + '_th_in'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_in_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        tops=[],
        data=[np.array(1.)],
        subgraph=X.subgraph
    )
    new_Xs.append(var_layer)

    quant_in_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',
        'axis': 1 if X.attrs['data_layout'] == 'NCHW' else 3,
        'mse_opt_num': mse_opt_num
    }
    quant_in_layer = defaultXLayer()
    quant_in_layer = quant_in_layer._replace(
        type=['MSEQuantize'],
        name=X.name + '_quantize_in',
        shapes=bottom_Xs[0].shapes[:],
        attrs=quant_in_attrs,
        bottoms=[new_xlayer_names[X.bottoms[0]], th_in_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(quant_in_layer)

    # KERNEL
    k_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    k_in_X = defaultXLayer()
    k_in_X = k_in_X._replace(
        type=['Input'],
        name=kernel_name,
        shapes=list(W.shape),
        bottoms=[],
        tops=[],
        subgraph=X.subgraph
        # TODO attrs
    )
    new_Xs.append(k_in_X)

    th_params_var_name = X.name + '_th_params'
    th_params_var_attrs = {
        # 'init_value': np.ones(W.shape[0]),  # OIHW
        'dtype': 'float32'
    }
    th_params_var_layer = defaultXLayer()
    th_params_var_layer = th_params_var_layer._replace(
        type=['Variable'],
        name=th_params_var_name,
        shapes=[list(W.shape)[0]],  # NCHW,
        attrs=th_params_var_attrs,
        bottoms=[],
        tops=[],
        data=[np.ones(W.shape[0])],
        subgraph=X.subgraph
    )
    new_Xs.append(th_params_var_layer)

    k_quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO: input types?
        'axis': 0,  # TODO: OIHW
        'mse_opt_num': mse_opt_num
    }
    k_quant_X = defaultXLayer()
    k_quant_X = k_quant_X._replace(
        type=['MSEQuantize'],
        name=kernel_name + '_quantize',
        shapes=list(W.shape),
        attrs=k_quant_attrs,
        bottoms=[kernel_name, th_params_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(k_quant_X)

    # BIAS
    b_in_attrs = {
        'dtype': 'float32',
        'layout': 'None'
    }
    b_in_X = defaultXLayer()
    b_in_X = b_in_X._replace(
        type=['Input'],
        name=bias_name,
        shapes=list(B.shape),
        bottoms=[],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(b_in_X)

    b_quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO: input types?
        'axis': 0  # TODO: NCHW
    }
    b_quant_X = defaultXLayer()
    b_quant_X = b_quant_X._replace(
        type=['MSEQuantizeBias'],
        name=bias_name + '_quantize',
        shapes=list(B.shape),
        attrs=b_quant_attrs,
        bottoms=[bias_name, th_in_var_name, th_params_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(b_quant_X)

    X = X._replace(
        bottoms=[X.name + '_quantize_in'] +
        [kernel_name + '_quantize', bias_name + '_quantize']
    )
    new_Xs.append(X)

    # Threshold layer
    th_out_var_name = X.name + '_th_out'
    var_attrs = {
        # 'init_value': np.array(1.),
        'dtype': 'float32'
    }
    var_layer = defaultXLayer()
    var_layer = var_layer._replace(
        type=['Variable'],
        name=th_out_var_name,
        shapes=[1],
        attrs=var_attrs,
        bottoms=[],
        tops=[],
        data=[np.array(1.)],
        subgraph=X.subgraph
    )
    # variable_layers.add(th_in_var_name)
    new_Xs.append(var_layer)

    # Quantize layer
    quant_attrs = {
        'quant_bitwidth': bitwidth,
        'dtype': 'float32',  # TODO??
        'axis': 1 if X.attrs['data_layout'] == 'NCHW' else 3,
        'mse_opt_num': mse_opt_num
    }
    quant_X = defaultXLayer()
    quant_X = quant_X._replace(
        type=['MSEQuantize'],
        name=X.name + "_quantize",
        shapes=X.shapes,
        attrs=quant_attrs,
        bottoms=[X.name, th_out_var_name],
        tops=[],
        subgraph=X.subgraph
    )
    new_Xs.append(quant_X)

    return new_Xs


# def get_input_quantize_layers(bottom_Xs, X, top_Xs, mse_opt_num, bitwidth,
#                               new_xlayer_names):
#     # type: (Dict[str, XLayer], XLayer,
#     #   Dict[str, XLayer])
#     #   -> List[XLayer]
#     """
#     TODO: Make more modular
#     """
#     new_Xs = [X]

#     # Threshold layer
#     th_out_var_name = X.name + '_th_out'
#     var_attrs = {
#         'init_value': np.array(1.),
#         'dtype': 'float32'
#     }
#     var_layer = defaultXLayer()
#     var_layer = var_layer._replace(
#         type = ['Variable'],
#         name = th_out_var_name,
#         shapes = [1],
#         attrs = var_attrs,
#         bottoms = [],
#         tops = []
#     )
#     #variable_layers.add(th_in_var_name)
#     new_Xs.append(var_layer)

#     # Quantize layer
#     quant_attrs = {
#         'quant_params': {
#             'bitwidth': 8
#         },
#         'dtype': 'float32', # TODO??
#         'axis': 1, #TODO NCHW
#         'mse_opt_num': mse_opt_num
#     }
#     quant_X = defaultXLayer()
#     quant_X = quant_X._replace(
#         type = ['MSEQuantize'],
#         name = X.name + "_quantize",
#         shapes = X.shapes,
#         attrs = quant_attrs,
#         bottoms = [X.name, th_out_var_name],
#         tops = []
#     )
#     new_Xs.append(quant_X)

#     return new_Xs

ADD_THRESHOLD_OPT_LAYER = {
    'Concat': get_concat_quantize_layers,
    'Scale': get_scale_quantize_layers,
    'Pooling': get_pooling_quantize_layers,
    'Eltwise': get_eltwise_quantize_layers,
    'Convolution': get_convolution_quantize_layers,
    # 'Input': get_input_quantize_layers
}


class XGraphPassAddMSEQuantLayers(XGraphBasePass):

    """
    Responsible for inserting MSE threshold optimization layers into the
    provided graph

    Attributes
    ----------

    """
    def __init__(self,
                 bitwidth=8,
                 mse_opt_num=50,
                 subgraphs_only=True,
                 last_opt_layer=None,
                 name='XGraphQuantPass',
                 output_png=None):
        super(XGraphPassAddMSEQuantLayers, self).__init__(
            name=name,
            output_png=output_png)

        self.bitwidth = bitwidth
        self.mse_opt_num = mse_opt_num
        self.subgraphs_only = subgraphs_only
        self.last_opt_layer = last_opt_layer

    def execute(self, xgraph):
        # type: (XGraph) -> XGraph
        """ """

        mse_opt_num = self.mse_opt_num

        skip = False
        new_xlayer_names = {}

        def add_quantization_layers(bottom_Xs, X, top_Xs):
            # type: (List[XLayer], XLayer, List[XLayer]) -> List[XLayer]
            """
            Replace the provided parameters layer with a list of xlayers
            adding threshold training layers before and after specific layers
            """
            # TODO: tops
            nonlocal skip, mse_opt_num, new_xlayer_names

            new_Xs = []
            # Keep track of changed layer names because layers are replaced by
            #   a combination of new layers

            if not skip and X.type[0] in ADD_THRESHOLD_OPT_LAYER and\
                    (not self.subgraphs_only or X.subgraph is not None):

                logger.debug("Add quant layers for: {}".format(X.name))
                xlayers = ADD_THRESHOLD_OPT_LAYER[X.type[0]](
                    bottom_Xs, X, top_Xs, mse_opt_num, self.bitwidth,
                    new_xlayer_names)

                new_Xs.extend(xlayers)
                new_xlayer_names[X.name] = xlayers[-1].name

                logger.debug("Number of new layers: {}".format(len(xlayers)))
            else:
                new_Xs.append(X)
                new_xlayer_names[X.name] = X.name

            if self.last_opt_layer is not None and\
                    self.last_opt_layer == X.name:
                # Start skipping adding threshold optimization layers from the
                #   next layer
                skip = True

            return new_Xs

        # Used for merging Variable layers that should point to the same
        #    threshold variable
        replace_bottoms = {}

        def merge_quantize_layers(bottom_Xs, X, top_Xs):
            # type: (List[XLayer], XLayer, List[XLayer]) -> List[XLayer]
            """
            Merge the provided parameters layer with preceding or succeeding
            layer if they cancel eachother out

            TODO: Formalize ParametersLayer comparison in a separate wrapper
                class?
            TODO: Remove quantize inter layers that are doing the identity
                operation
            """
            nonlocal replace_bottoms

            def get_replace_bottom(key):
                if key not in replace_bottoms or replace_bottoms[key] == key:
                    return key
                return get_replace_bottom(replace_bottoms[key])

            new_Xs = []
            # TODO: Merge MaxPool th_in with th_out
            if 'Pooling' in X.type and X.attrs['pool_type'] == 'Max' and \
                    'MSEMockQuantize' in top_Xs[0].type:
                # assert()
                assert(bottom_Xs[0].type[0] in
                       ['MSEQuantize', 'MSEMockQuantize'])
                logger.debug("Layer is a MaxPool layer with preceding and"
                             " suceeding MSEQuantize layers")
                # Note: MaxPool th_out === th_in

                # top is a ThresholdTrainingQuantizeInter layer
                th_out_var = top_Xs[0].bottoms[1]
                th_in_var = get_replace_bottom(bottom_Xs[0].bottoms[1])
                replace_bottoms[th_out_var] = th_in_var
                logger.debug("Point: {} -> {}".format(th_out_var, th_in_var))
                new_Xs.append(X)

            elif 'MSEQuantize' in X.type and 'remove_layer' in X.attrs:
                assert(len(bottom_Xs) in [2, 4])
                logger.debug("Layer is a MSEQuantize layer to be removed")
                # If this quantize layer is indicated to be removed by previous
                #  layer, then remove this quantize layer
                if not X.attrs['remove_layer']:
                    logger.debug("Don't remove layer")
                    new_Xs.append(X)

            elif 'MSEQuantize' in X.type or 'MSEMockQuantize' in X.type:
                # Check if all bottom is also a quantization layer
                assert(len(bottom_Xs) in [2, 4])

                new_Xs.append(X)

                logger.debug("Layer is an MSEQuantize layer")
                th_in_var_name = X.bottoms[1]

                for i in range(len(top_Xs)):
                    top_X = top_Xs[i]
                    t_attrs, attrs = top_Xs[i].attrs, X.attrs
                    logger.debug("-- top_X: {}, type: {}"
                                 .format(top_Xs[i].name, top_Xs[i].type))
                    # logger.debug(t_attrs)
                    # logger.debug(attrs)
                    if 'MSEQuantize' in top_X.type:
                        # remove top later on
                        top_X.attrs['remove_layer'] = True

                        # Refer top layer variable threshold inputs to this
                        #   layer's variable threshold input. If the
                        #   top_X_in_var_name is already set to be replaced,
                        #   then we don't want to set this again
                        #   because it might screw up the topological order
                        top_X_th_in_var_name = top_X.bottoms[1]
                        if top_X_th_in_var_name not in replace_bottoms:
                            logger.debug("Point top_X input var: {} -> {}"
                                         .format(top_X_th_in_var_name,
                                                 th_in_var_name))
                            replace_bottoms[top_X_th_in_var_name] \
                                = th_in_var_name

            elif 'Variable' in X.type and X.name in replace_bottoms:
                logger.debug("Replace VariableLayer: {} with {}"
                             .format(X.name, replace_bottoms[X.name]))
                # TODO get_replace_bottom(...)
                for top_X in top_Xs:
                    for i, b in enumerate(top_X.bottoms):
                        if b == X.name:
                            logger.debug("top_X: {}, idx: {}"
                                         .format(top_X.name, i))
                            top_X.bottoms[i] = replace_bottoms[b]
                            logger.debug(top_X.bottoms)
                # Remove layer
            else:
                X = X._replace(
                    tops=[]
                )
                new_Xs.append(X)

            return new_Xs

        # Here we call the actual graph passes
        # We do two of them
        # 1. A forward pass which adds all threshold training layers
        # 2. A forward pass to merge quant training layers

        # 1.
        fancy_logger.banner("GRAPH PASS ADD QUANT THRESHOLD OPTIMIZATION"
                            " LAYERS")

        output_png = None if self.output_png is None else \
            self.output_png.split('.')[0] + '_add.' + \
            self.output_png.split('.')[1]
        xgraph = self._replace_layer_pass(
            xgraph,
            add_quantization_layers,
            name=self.name,
            output_png=output_png
        )

        # 2.
        fancy_logger.banner("GRAPH PASS MERGE QUANT LAYERS")

        output_png = None if self.output_png is None else \
            self.output_png.split('.')[0] + '_merge.' + \
            self.output_png.split('.')[1]
        xgraph = self._replace_layer_pass(
            xgraph,
            merge_quantize_layers,
            name=self.name,
            output_png=output_png
        )

        return xgraph
