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
Module for transforming ONNX L2 operators to XLayer objects

L2: Convolution related operators
"""

import math
import logging
import numpy as np
import pyxir as px

from typing import Dict, List

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper
from .tools import eltwise_any_op

logger = logging.getLogger('pyxir')


@register_onnx_2_xlayer_converter("AveragePool")
def avg_pool(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX AveragePool to XLayer Pooling (Avg) conversion function"""
    logger.info("ONNX AveragePool -> XLayer Pooling (Avg)")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    auto_pad = node_attrs['auto_pad'] if 'auto_pad' in node_attrs\
        else 'NOTSET'
    ceil_mode = bool(node_attrs['ceil_mode']) if 'ceil_mode' in node_attrs\
        else False
    count_include_pad = node_attrs['count_include_pad']\
        if 'count_include_pad' in node_attrs else 0
    kernel_shape = node_attrs['kernel_shape'] if 'kernel_shape' in node_attrs\
        else W.shape[2:]
    kernel_h, kernel_w = kernel_shape
    pads = node_attrs['pads'] if 'pads' in node_attrs\
        else None
    strides = node_attrs['strides'] if 'strides' in node_attrs\
        else [1, 1]
    stride_h, stride_w = strides

    if auto_pad not in ['NOTSET', "SAME_UPPER", "SAME_LOWER"]:
        raise ValueError("AveragePool autopad attribute not supported but was:"
                         " {}".format(auto_pad))

    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        out_h, out_w = int(math.ceil(in_h / stride_h)), int(math.ceil(in_w / stride_w))
        pad_h = (out_h - 1) * stride_h + kernel_h - in_h
        pad_w = (out_w - 1) * stride_w + kernel_w - in_w
        if auto_pad == "SAME_UPPER":
            pad_ht, pad_hb = pad_h // 2, pad_h - (pad_h // 2)
            pad_wl, pad_wr = pad_w // 2, pad_w - (pad_w // 2)
        else:
            pad_ht, pad_hb = pad_h - (pad_h // 2), pad_h // 2
            pad_wl, pad_wr = pad_w - (pad_w // 2), pad_w // 2
        padding = [pad_ht, pad_hb, pad_wl, pad_wr]
    else:
        padding = pads if pads is not None else [0, 0, 0, 0]

    # [pad_ht, pad_hb, pad_wl, pad_wr] -> [pad_ht, pad_wl,  pad_hb, pad_wr]
    # TODO move internal pool padding to [pad_ht, pad_hb, pad_wl, pad_wr]
    padding = [padding[i] for i in [0, 2, 1, 3]]

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    X = px.ops.pool2d(
        op_name=px.stringify(name),
        input_layer=iX,
        pool_type='Avg',
        pool_size=kernel_shape,
        strides=strides,
        padding=padding,
        layout='NCHW',
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Conv")
def conv(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Conv to XLayer Conv conversion function"""
    logger.info("ONNX Conv -> XLayer Conv (+ BiasAdd)")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    W_name = bottoms[1]
    wX = xmap[W_name]  # OIHW

    B_name = bottoms[2] if len(bottoms) == 3 else None
    bX = xmap[B_name] if len(bottoms) == 3 else None

    auto_pad = node_attrs['auto_pad'] if 'auto_pad' in node_attrs\
        else 'NOTSET'
    dilations = node_attrs['dilations'] if 'dilations' in node_attrs\
        else [1, 1]
    dil_h, dil_w = dilations
    groups = node_attrs['group'] if 'group' in node_attrs\
        else 1
    kernel_shape = node_attrs['kernel_shape'] if 'kernel_shape' in node_attrs\
        else wX.shapes[2:]
    kernel_h, kernel_w = kernel_shape
    pads = node_attrs['pads'] if 'pads' in node_attrs\
        else None
    strides = node_attrs['strides'] if 'strides' in node_attrs\
        else [1, 1]
    stride_h, stride_w = strides

    channels = wX.shapes[0]
    assert wX.shapes[1] == in_c // groups

    assert auto_pad == 'NOTSET' or pads is None
    if (auto_pad == 'NOTSET' and pads is None) or auto_pad == 'VALID':
        padding = [0, 0, 0, 0]  # ht, hb, wl, wr
    elif auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        out_h, out_w = int(math.ceil(in_h / stride_h)), int(math.ceil(in_w / stride_w))
        pad_h = (out_h - 1) * stride_h + (dil_h * (kernel_h - 1) + 1) - in_h
        pad_w = (out_w - 1) * stride_w + (dil_w * (kernel_w - 1) + 1) - in_w
        if auto_pad == "SAME_UPPER":
            pad_ht, pad_hb = pad_h // 2, pad_h - (pad_h // 2)
            pad_wl, pad_wr = pad_w // 2, pad_w - (pad_w // 2)
        else:
            pad_ht, pad_hb = pad_h - (pad_h // 2), pad_h // 2
            pad_wl, pad_wr = pad_w - (pad_w // 2), pad_w // 2
        padding = [pad_ht, pad_hb, pad_wl, pad_wr]
    else:
        assert len(pads) % 2 == 0
        half = len(pads) // 2
        padding = []
        for i in range(half):
            padding.extend([pads[i], pads[i+half]])

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant_weights = node_attrs['vai_quant_weights']\
        if 'vai_quant_weights' in node_attrs else []
    vai_quant_biases = node_attrs['vai_quant_biases']\
        if 'vai_quant_biases' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    conv_name = name if B_name is None else name + '_Conv'
    X = px.ops.conv2d(
        op_name=px.stringify(conv_name),
        input_layer=iX,
        weights_layer=wX,
        kernel_size=kernel_shape,
        strides=strides,
        padding_hw=padding,
        dilation=dilations,
        groups=groups,
        channels=channels,
        data_layout='NCHW',
        kernel_layout='OIHW',
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        vai_quant_weights=vai_quant_weights,
        vai_quant_biases=vai_quant_biases,
        onnx_id=name
    )
    res = [X]

    if B_name is not None:
        bias_add_X = xlf.get_xop_factory_func('BiasAdd')(
            op_name=px.stringify(name),
            axis=1,
            input_layer=X,
            bias_layer=bX,
            onnx_id=name
        )

        res.append(bias_add_X)

    return res


@register_onnx_2_xlayer_converter("ConvInteger")
def conv_integer(node: NodeWrapper,
                 params: Dict[str, np.ndarray],
                 xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Convinteger to XLayer Conv conversion function"""
    logger.info("ONNX ConvInteger -> XLayer Conv")
    return conv(node, params, xmap)


@register_onnx_2_xlayer_converter("ConvTranspose")
def conv_transpose(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX ConvTranspose to XLayer Conv2DTranspose conversion function"""
    logger.info("ONNX ConvTranspose -> XLayer Conv2DTranspose (+ BiasAdd)")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    W_name = bottoms[1]
    wX = xmap[W_name]  # OIHW
    assert wX.shapes[1] == in_c

    B_name = bottoms[2] if len(bottoms) == 3 else None
    bX = xmap[B_name] if len(bottoms) == 3 else None

    auto_pad = node_attrs['auto_pad'] if 'auto_pad' in node_attrs\
        else 'NOTSET'
    dilations = node_attrs['dilations'] if 'dilations' in node_attrs\
        else [1, 1]
    dil_h, dil_w = dilations
    groups = node_attrs['group'] if 'group' in node_attrs\
        else 1
    kernel_shape = node_attrs['kernel_shape'] if 'kernel_shape' in node_attrs\
        else wX.shapes[2:]
    kernel_h, kernel_w = kernel_shape
    output_padding = node_attrs['output_padding'] \
        if 'output_padding' in node_attrs else [0, 0]
    if np.sum(output_padding) != 0:
        raise NotImplementedError("Conv2DTranspose with output padding not"
                                  " equal to a zero vector is unsupported")
    out_pad_h, out_pad_w = output_padding
    output_shape = node_attrs['output_shape'] if 'output_shape' in node_attrs\
        else None
    pads = node_attrs['pads'] if 'pads' in node_attrs\
        else None
    strides = node_attrs['strides'] if 'strides' in node_attrs\
        else [1, 1]
    stride_h, stride_w = strides

    channels = wX.shapes[0]

    if output_shape is None:
        assert auto_pad == 'NOTSET' or pads is None
        if (auto_pad == 'NOTSET' and pads is None) or auto_pad == 'VALID':
            padding = [0, 0, 0, 0]  # ht, hb, wl, wr
        elif auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
            out_h, out_w = in_h * stride_h, in_w * stride_w
            pad_h = stride_h * (in_h - 1) + out_pad_h + ((kernel_h - 1) * dil_h + 1) - out_h
            pad_w = stride_w * (in_w - 1) + out_pad_w + ((kernel_w - 1) * dil_w + 1) - out_w
            if auto_pad == "SAME_UPPER":
                pad_ht, pad_hb = pad_h // 2, pad_h - (pad_h // 2)
                pad_wl, pad_wr = pad_w // 2, pad_w - (pad_w // 2)
            else:
                pad_ht, pad_hb = pad_h - (pad_h // 2), pad_h // 2
                pad_wl, pad_wr = pad_w - (pad_w // 2), pad_w // 2
            padding = [pad_ht, pad_hb, pad_wl, pad_wr]
        else:
            padding = pads
    else:
        out_h, out_w = output_shape[2], output_shape[3]
        pad_h = stride_h * (in_h - 1) + out_pad_h + ((kernel_h - 1) * dil_h + 1) - out_h
        pad_w = stride_w * (in_w - 1) + out_pad_w + ((kernel_w - 1) * dil_w + 1) - out_w

        if auto_pad != 'SAME_UPPER':
            pad_ht, pad_hb = pad_h // 2, pad_h - (pad_h // 2)
            pad_wl, pad_wr = pad_w // 2, pad_w - (pad_w // 2)
        else:
            pad_ht, pad_hb = pad_h - (pad_h // 2), pad_h // 2
            pad_wl, pad_wr = pad_w - (pad_w // 2), pad_w // 2
        padding = [pad_ht, pad_hb, pad_wl, pad_wr]

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant_weights = node_attrs['vai_quant_weights']\
        if 'vai_quant_weights' in node_attrs else []
    vai_quant_biases = node_attrs['vai_quant_biases']\
        if 'vai_quant_biases' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    conv_name = name if B_name is None else name + '_Conv'
    X = px.ops.conv2d_transpose(
        op_name=px.stringify(conv_name),
        input_layer=iX,
        weights_layer=wX,
        kernel_size=kernel_shape,
        strides=strides,
        padding_hw=padding,
        dilation=dilations,
        groups=groups,
        channels=channels,
        data_layout='NCHW',
        kernel_layout='OIHW',
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        vai_quant_weights=vai_quant_weights,
        vai_quant_biases=vai_quant_biases,
        onnx_id=name
    )
    res = [X]

    if B_name is not None:
        bias_add_X = xlf.get_xop_factory_func('BiasAdd')(
            op_name=px.stringify(name),
            axis=1,
            input_layer=X,
            bias_layer=bX,
            onnx_id=name
        )

        res.append(bias_add_X)

    return res


@register_onnx_2_xlayer_converter("Flatten")
def flatten(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]) -> List[XLayer]:
    """
    ONNX Flatten to XLayer Flatten or Reshape conversion function

    ONNX: Flattens the input tensor into a 2D matrix. If input tensor has
    shape (d_0, d_1, ... d_n) then the output will have shape
    (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis+1) ... X dn).
    See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten
    """
    logger.info("ONNX Flatten -> XLayer Flatten/Reshape")

    assert len(node.get_outputs()) == 1
    assert len(node.get_inputs()) == 1

    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    shape = iX.shapes.tolist()
    rank = len(shape)

    axis = node_attrs['axis'] if 'axis' in node_attrs else 1
    assert axis >= -rank and axis <= rank

    if axis == 1 or axis == -(rank-1):
        X = px.ops.batch_flatten(px.stringify(name), [iX], onnx_id=name)
    else:
        shape_1 = int(np.prod(shape[:axis])) if shape[:axis] != [] else 1
        shape_2 = int(np.prod(shape[axis:])) if shape[axis:] != [] else 1

        newshape = [shape_1, shape_2]

        X = px.ops.reshape(
            op_name=px.stringify(name),
            newshape=newshape,
            input_layer=iX,
            onnx_id=name
        )

    return [X]


@register_onnx_2_xlayer_converter("GlobalAveragePool")
def global_avg_pool(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX GlobalAveragePool to XLayer Pooling (Avg) conversion function"""
    logger.info("ONNX GlobalAveragePool -> XLayer Pooling (Avg)")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    X = xlf.get_xop_factory_func('GlobalPooling')(
        op_name=px.stringify(name),
        input_layer=iX,
        pool_type='Avg',
        layout='NCHW',
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("GlobalMaxPool")
def global_max_pool(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX GlobalMaxPool to XLayer Pooling (Max) conversion function"""
    logger.info("ONNX GlobalMaxPool -> XLayer Pooling (Max)")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    X = xlf.get_xop_factory_func('GlobalPooling')(
        op_name=px.stringify(name),
        input_layer=iX,
        pool_type='Max',
        layout='NCHW',
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("LRN")
def lrn(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    return eltwise_any_op("LRN", node, params, xmap)


@register_onnx_2_xlayer_converter("MaxPool")
def max_pool(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    """ONNX MaxPool to XLayer MaxPool conversion function"""
    logger.info("ONNX MaxPool -> XLayer Pooling")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    auto_pad = node_attrs['auto_pad'] if 'auto_pad' in node_attrs\
        else 'NOTSET'
    ceil_mode = bool(node_attrs['ceil_mode']) if 'ceil_mode' in node_attrs\
        else False
    dilations = node_attrs['dilations'] if 'dilations' in node_attrs\
        else [1, 1]
    dil_h, dil_w = dilations
    kernel_shape = node_attrs['kernel_shape'] if 'kernel_shape' in node_attrs\
        else W.shape[2:]
    kernel_h, kernel_w = kernel_shape
    pads = node_attrs['pads'] if 'pads' in node_attrs\
        else None
    storage_order = node_attrs['storage_order']\
        if 'storage_order' in node_attrs else 0
    strides = node_attrs['strides'] if 'strides' in node_attrs\
        else [1, 1]
    stride_h, stride_w = strides

    if auto_pad not in ['NOTSET', 'VALID', 'SAME_UPPER', 'SAME_LOWER']:
        raise ValueError("MaxPool autopad attribute not supported but was: {}"
                         .format(auto_pad))
    if storage_order != 0:
        raise ValueError("MaxPool storage_order != 0 attribute not supported"
                         " but got: {}".format(storage_order))
    # TODO dilations
    if dilations != [1, 1]:
        raise NotImplementedError("Dilations are expected to be [1, 1] for"
                                  " now")

    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        out_h, out_w = int(math.ceil(in_h / stride_h)), int(math.ceil(in_w / stride_w))
        pad_h = (out_h - 1) * stride_h + (dil_h * (kernel_h - 1) + 1) - in_h
        pad_w = (out_w - 1) * stride_w + (dil_w * (kernel_w - 1) + 1) - in_w
        if auto_pad == "SAME_UPPER":
            pad_ht, pad_hb = pad_h // 2, pad_h - (pad_h // 2)
            pad_wl, pad_wr = pad_w // 2, pad_w - (pad_w // 2)
        else:
            pad_ht, pad_hb = pad_h - (pad_h // 2), pad_h // 2
            pad_wl, pad_wr = pad_w - (pad_w // 2), pad_w // 2
        padding = [pad_ht, pad_hb, pad_wl, pad_wr]
    else:
        padding = pads if pads is not None else [0, 0, 0, 0]

    # [pad_ht, pad_hb, pad_wl, pad_wr] -> [pad_ht, pad_wl,  pad_hb, pad_wr]
    # TODO move internal pool padding to [pad_ht, pad_hb, pad_wl, pad_wr]
    padding = [padding[i] for i in [0, 2, 1, 3]]

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    X = px.ops.pool2d(
        op_name=px.stringify(name),
        input_layer=iX,
        pool_type='Max',
        pool_size=kernel_shape,
        strides=strides,
        padding=padding,
        layout='NCHW',
        ceil_mode=ceil_mode,
        count_include_pad=False,
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("MaxRoiPool")
def max_roi_pool(node: NodeWrapper,
                 params: Dict[str, np.ndarray],
                 xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX MaxRoiPool to XLayer AnyOp conversion function"""
    logger.info("ONNX MaxRoiPool -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes
    rois = xmap[bottoms[1]]
    num_rois = rois.shapes[0]

    out_h, out_w = [int(i) for i in node_attrs['pooled_shape']]

    out_shape = [num_rois, in_c, out_h, out_w]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("MaxUnPool")
def max_unpool(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX MaxUnPool to XLayer AnyOp conversion function"""
    logger.info("ONNX MaxPool -> XLayer Pooling")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    in_b, in_c, in_h, in_w = iX.shapes

    if len(bottoms) == 3:
        out_shape = [int(i) for i in list(xmap[bottoms[2]].data[0])]
    else:
        kernel_shape = node_attrs['kernel_shape'] \
            if 'kernel_shape' in node_attrs else W.shape[2:]
        kernel_h, kernel_w = kernel_shape
        pads = node_attrs['pads'] if 'pads' in node_attrs\
            else None
        strides = node_attrs['strides'] if 'strides' in node_attrs\
            else [1, 1]
        stride_h, stride_w = strides

        padding = pads if pads is not None else [0, 0, 0, 0]

        # [pad_ht, pad_hb, pad_wl, pad_wr] -> [pad_ht, pad_wl,  pad_hb, pad_wr]
        # TODO move internal pool padding to [pad_ht, pad_hb, pad_wl, pad_wr]
        padding = [padding[i] for i in [0, 2, 1, 3]]
        pad_ht, pad_wl, pad_hb, pad_wr = padding

        out_h = (in_h - 1) * stride_h + kernel_h - pad_ht - pad_hb
        out_w = (in_w - 1) * stride_w + kernel_w - pad_wl - pad_wr
        out_shape = [in_b, in_c, out_h, out_w]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Pad")
def pad(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Pad to XLayer Pad conversion function"""
    logger.info("ONNX Pad -> XLayer Pad")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    if len(bottoms) > 1:
        padding = [int(i) for i in xmap[bottoms[1]].data[0]]
        pad_value = float(xmap[bottoms[2]].data[0])
    else:
        pad_str = 'pads' if 'pads' in node_attrs else 'paddings'
        padding = [int(i) for i in node_attrs[pad_str]]
        pad_value = float(node_attrs['value']) \
            if 'value' in node_attrs else 0.

    h = len(padding) // 2
    padding = [[padding[i], padding[i + h]] for i in range(h)]

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    X = px.ops.pad(
        op_name=px.stringify(name),
        input_layer=iX,
        padding=padding,
        pad_value=pad_value,
        onnx_id=name,
        vai_quant=vai_quant,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
    )

    return [X]


@register_onnx_2_xlayer_converter("QLinearConv")
def qlinearconv(node: NodeWrapper,
                params: Dict[str, np.ndarray],
                xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX QLinearConv to XLayer AnyOp conversion function"""
    raise NotImplementedError("Unsupported ONNX QLinearConv operator")


@register_onnx_2_xlayer_converter("Upsample")
def upsample(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Upsample to XLayer Upsampling2D conversion function"""

    logger.info("ONNX Upsample -> XLayer Upsampling2D")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    assert len(bottoms) == 2 or 'scales' in node_attrs

    iX = xmap[bottoms[0]]  # NCHW

    scales = [float(i) for i in (list(xmap[bottoms[1]].data[0])
              if 'scales' not in node_attrs
              else node_attrs['scales'])]
    assert len(scales) == len(iX.shapes)
    scale_n, scale_c, scale_h, scale_w = scales

    if scale_n != 1:
        raise NotImplementedError("Unsupported upsampling layer with scale"
                                  " for batch dim != 1")
    if scale_c != 1:
        raise NotImplementedError("Unsupported upsampling layer with scale"
                                  " for channel dim != 1")

    mode = node_attrs['mode'] if 'mode' in node_attrs \
        else 'nearest'
    if mode == 'nearest':
        mode = 'nearest_neighbor'

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    X = px.ops.upsampling2d(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        scale_h=scale_h,
        scale_w=scale_w,
        data_layout='NCHW',
        method=mode,
        onnx_id=name
    )

    return [X]
