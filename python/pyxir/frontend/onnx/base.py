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
Module for transforming ONNX graph to XGraph representation


"""

import os
import onnx
import logging
import pyxir as px

from onnx import numpy_helper

from pyxir.type import TypeCode
from pyxir.opaque_func_registry import register_opaque_func
from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph import XGraph
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.algorithms.topological_sorting import sort_topologically

from .onnx_io import load_onnx_model_from_file
from .onnx_tools import NodeWrapper, TensorTypeWrapper,\
    get_onnx_elem_type_2_dtype
from .onnx_2_xlayer_registry import ONNX2XLayerRegistry
from .ops import onnx_l0_other, onnx_l1_basic_nn, onnx_l2_convolution, \
    onnx_l3_math_and_transform, onnx_l4_broadcast_and_reductions, \
    onnx_l5_vision, onnx_l6_recurrent, onnx_l11_quantization

logger = logging.getLogger("pyxir")


def from_onnx(onnx_model, postprocessing=None):
    """ Wrapper around internal ONNX model import function """
    return _from_onnx(onnx_model, postprocessing=postprocessing)


@register_opaque_func('pyxir.onnx.from_onnx', [TypeCode.XGraph, TypeCode.Str])
def from_onnx_opaque_func(xgraph, arg):
    """ Expose the ONNX model import function as an opaque function
        so it can be called from both Python and C++ """

    onnx_model = load_onnx_model_from_file(arg)

    _from_onnx(onnx_model, xgraph=xgraph)


@register_opaque_func('pyxir.onnx.from_onnx_bytes', [TypeCode.XGraph,
                                                     TypeCode.Byte])
def from_onnx_bytes_opaque_func(xgraph, arg):
    """ Expose the ONNX model import function as an opaque function
        so it can be called from both Python and C++ """

    onnx_model = onnx.load_model_from_string(arg)

    _from_onnx(onnx_model, xgraph=xgraph)


def _from_onnx(onnx_model, xgraph=None, postprocessing=None):
    # type: (onnx.onnx_ONNX_RELEASE_ml_pb2.ModelProto, XGraph, List[str])
    #   -> XGraph
    """
    Tranform ONNX model into XGraph

    Arguments
    ---------
    onnx_model: onnx.onnx_ONNX_RELEASE_ml_pb2.ModelProto
        The ONNX model to be transformed into a XGraph
    xgraph: XGraph (Optional)
        The XGraph object to be used for string the transformed ONNX model
    postprocessing: List[str] (Optional)
        a list of postprocessing layers to be added

    Returns
    -------
    xgraph: XGraph
        the created xgraph model
    """

    onnx_graph = onnx_model.graph

    if xgraph is None:
        xgraph = XGraph(name=onnx_graph.name)
    else:
        xgraph.set_name(onnx_graph.name)

    if postprocessing is None:
        postprocessing = []

    onnx_elem_type_2_dtype = get_onnx_elem_type_2_dtype()
    registry = ONNX2XLayerRegistry()
    xgraph_factory = XGraphFactory()

    # Metadata
    quant_info = {}
    for meta in onnx_model.metadata_props:
        meta_key_split = meta.key.split("--")
        if meta_key_split[0] == "vitis_ai_quant":
            qkey = meta_key_split[-1]
            if qkey not in quant_info:
                quant_info[qkey] = {}
            quant_info[qkey][meta_key_split[1]] = meta.value
    quant_keys = list(quant_info.keys())
    xgraph.meta_attrs['quant_keys'] = quant_keys
    for qkey in quant_keys:
        xgraph.meta_attrs[qkey] = quant_info[qkey]

    params = {e.name: numpy_helper.to_array(e)
              for e in onnx_model.graph.initializer}
    logger.debug("ONNX params size: {}".format(len(params)))

    net = []
    xmap = {}

    # Setup parameters layers
    for p_name in params.keys():
        # logger.debug("pyxir.onnx param: {}".format(p_name))
        # if p_name not in onnx_graph.input:
        cX = xlf.get_xop_factory_func('Constant')(
            op_name=px.stringify(p_name),
            value=params[p_name],
            onnx_id=p_name
        )

        # xmap[cX.name] = cX
        xmap[cX.attrs['onnx_id']] = cX

    # Setup input xlayers
    for input_proto in onnx_graph.input:
        name = input_proto.name
        # logger.debug("pyxir.onnx input: {}".format(name))
        if name not in params:
            logger.debug("input_proto: {}".format(name))
            t_type = TensorTypeWrapper(input_proto.type.tensor_type)
            dtype = t_type.get_dtype()
            shape = t_type.get_shape()
            X = xlf.get_xop_factory_func('Input')(
                px.stringify(name),
                list(shape),
                dtype=dtype,
                onnx_id=name
            )

            net.append(X)
            # xmap[X.name] = X
            xmap[X.attrs['onnx_id']] = X

    for node in onnx_graph.node:
        # logger.debug("pyxir.onnx node: {}".format(node))
        wrapped_node = NodeWrapper(node)
        op_type = wrapped_node.get_op_type()
        Xs = registry[op_type](wrapped_node, params, xmap)
        net.extend(Xs)

    # Postprocessing
    OP_2_XLAYER = {
        'Softmax': xlf.get_xop_factory_func('Softmax',
                                            internal=True)
    }

    # Add additional output layers to the network that are not specified
    #   in the network file (usually only used for adding softmax layers)
    for i, output in enumerate(postprocessing):
        if output not in OP_2_XLAYER:
            continue
            # raise NotImplementedError(
            #     "The provided output operation: {} is invalid."
            #     " The valid output operations are: {}"
            #     .format(output, list(OP_2_XLAYER.keys())))
        op_name = output + str(i)

        # Update tops of current last layer
        X = net[-1]
        X.tops.append(op_name)
        X = OP_2_XLAYER[output](op_name, [X])

        if X.name in net:
            raise ValueError("This should never happen. Error because the"
                             " generated output name already exists in the"
                             " network dictionary used for setup.")

        net.append(X)
        xmap[X.name] = X

    # net = sort_topologically(list(xmap.values()))

    xgraph_factory.build_from_xlayer(
        net=net,
        xgraph=xgraph,
        name=onnx_graph.name,
        blobs=False
    )

    return xgraph


def prequantize_onnx_model(onnx_model, target, inputs_func, out_file,
                           **kwargs):

    xgraph = _from_onnx(onnx_model)

    xgraph = px.partition(xgraph, [target])
    xgraph = px.optimize(xgraph, target)

    q_xgraph = px.quantize(xgraph, target, inputs_func, **kwargs)

    # Move quant_info information from XGraph to ONNX model
    tensor_quant_info = {}
    for X in q_xgraph.get_layers():
        if "vai_quant" in X.attrs and X.attrs["vai_quant"] != []:
            tensor_name = X.attrs['onnx_id']
            if tensor_name in tensor_quant_info:
                raise NotImplementedError("Quantization for ONNX tensor: {}"
                                          " already provided. Merging of"
                                          " multiple tensor quantization info"
                                          " parameters not supported yet")
            tensor_quant_info[tensor_name] = {
                "vai_quant": X.attrs["vai_quant"]
            }

            for vai_quant_elem in X.attrs['vai_quant']:
                tensor_quant_info[tensor_name][vai_quant_elem] = \
                    X.attrs[vai_quant_elem]

    for node in onnx_model.graph.node:
        node_w = NodeWrapper(node)
        tensor_name = node_w.get_outputs()[0]
        if tensor_name in tensor_quant_info:
            node_w.add_attribute(
                'vai_quant',
                list(tensor_quant_info[tensor_name]['vai_quant']),
                'STRINGS'
            )
            for vai_quant_elem in tensor_quant_info[tensor_name]['vai_quant']:
                node_w.add_attribute(
                    vai_quant_elem,
                    tensor_quant_info[tensor_name][vai_quant_elem],
                    'INTS'
                )

    # q_output = q_xgraph.get_quantizer_output()

    # for qkey in q_output.keys():
    #     quant_file = q_output.get_q_file(qkey)
    #     quant_info_file = q_output.get_q_info(qkey)
    #     quant_orig_pb = q_output.get_orig_pb(qkey)

    #     if not os.path.isfile(quant_info_file):
    #         raise ValueError("quant file: {} for qkey: {} does not exist"
    #                          .format(quant_info_file, qkey))

    #     meta = onnx_model.metadata_props.add()
    #     meta.key = "vitis_ai_quant--q_file--" + qkey
    #     meta.value = str(quant_file)

    #     meta = onnx_model.metadata_props.add()
    #     meta.key = "vitis_ai_quant--q_info--" + qkey
    #     meta.value = str(quant_info_file)

    #     meta = onnx_model.metadata_props.add()
    #     meta.key = "vitis_ai_quant--orig_pb--" + qkey
    #     meta.value = str(quant_orig_pb)

    onnx.save(onnx_model, out_file)
