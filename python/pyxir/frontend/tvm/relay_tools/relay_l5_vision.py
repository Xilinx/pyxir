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
Module for transforming Relay L5 operators to XLayer objects

L5: Vision operators
"""

import math
import logging
import numpy as np
import pyxir as px

from typing import Dict, List, Callable

import tvm
from tvm.relay.expr import Expr

from pyxir import graph
from pyxir.graph.layer import XLayer
from pyxir.graph.layer import xlayer_factory as xlf

from .relay_2_xlayer_registry import (
    register_relay_2_xlayer_converter,
    register_relay_2_xlayer_converter_base,
)

logger = logging.getLogger("pyxir")


@register_relay_2_xlayer_converter_base("image.resize")
def image_resize(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Conversion of Relay 'image.resize' layer to upsampling2d layer

    Relay
    -----
    Type: tvm.relay.image.resize
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (relay.Expr)
            The input data tensor.
        - size (Tuple of Int or Expr)
            The out size to which the image will be resized.
        - layout (str, optional)
            Layout of the input.
        - method (str, optional)
            Scale method to used [nearest_neighbor, bilinear, bicubic].
        - coordinate_transformation_mode (string, optional)
            Describes how to transform the coordinate in the resized tensor to the coordinate
            in the original tensor. Refer to the ONNX Resize operator specification for details.
            [half_pixel, align_corners, asymmetric]
        - out_dtype (str, optional)
            Type to return. If left None returns the same type as input.
    """
    assert len(in_xlayers) == 1
    in_shape = in_xlayers[0].shapes[:]

    out_h, out_w = [int(e) for e in expr.attrs.size]
    layout = str(expr.attrs.layout)
    method = str(expr.attrs.method)
    coordinate_transformation_mode = str(expr.attrs.coordinate_transformation_mode)
    out_dtype = str(expr.attrs.out_dtype) if expr.attrs.out_dtype is not None else None

    h_index, w_index = layout.index("H"), layout.index("W")
    in_h, in_w = in_shape[h_index], in_shape[w_index]
    if coordinate_transformation_mode == "asymmetric":
        scale_h, scale_w = out_h / in_h, out_w / in_w
        X = px.ops.upsampling2d(
            op_name,
            in_xlayers,
            scale_h=scale_h,
            scale_w=scale_w,
            data_layout=layout,
            method=method,
            relay_id=[hash(expr)],
        )
    else:
        # AnyOp
        out_shape = in_shape.tolist()
        out_shape[h_index] = out_h
        out_shape[w_index] = out_w
        X = px.ops.any_op(
            op_name, in_xlayers, any_shape=out_shape[:], relay_id=[hash(expr)]
        )

    logger.debug("-- outshape: {}".format(list(X.shapes)))
    return X


@register_relay_2_xlayer_converter_base("image.resize2d")
def image_resize2d(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Conversion of Relay 'image.resize2d' layer to upsampling2d layer

    Relay
    -----
    Type: tvm.relay.image.resize2d
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (relay.Expr)
            The input data tensor.
        - size (Tuple of Int or Expr)
            The out size to which the image will be resized.
        - layout (str, optional)
            Layout of the input.
        - method (str, optional)
            Scale method to used [nearest_neighbor, bilinear, bicubic].
        - coordinate_transformation_mode (string, optional)
            Describes how to transform the coordinate in the resized tensor to the coordinate
            in the original tensor. Refer to the ONNX Resize operator specification for details.
            [half_pixel, align_corners, asymmetric]
        - rounding_method: string, optional
            indicates how to find the "nearest" pixel in nearest_neighbor method
            [round, floor, ceil]
        - cubic_alpha: float
            Spline Coefficient for bicubic interpolation
        - cubic_exclude: int
            Flag to exclude exterior of the image during bicubic interpolation
        - out_dtype (str, optional)
            Type to return. If left None returns the same type as input.
    """
    assert len(in_xlayers) == 1
    in_shape = in_xlayers[0].shapes[:]

    out_h, out_w = [int(e) for e in expr.attrs.size]
    layout = str(expr.attrs.layout)
    method = str(expr.attrs.method)
    coordinate_transformation_mode = str(expr.attrs.coordinate_transformation_mode)
    rounding_method = str(expr.attrs.rounding_method)
    cubic_alpha = float(expr.attrs.cubic_alpha)
    cubic_exclude = int(expr.attrs.cubic_exclude)

    out_dtype = str(expr.attrs.out_dtype) if expr.attrs.out_dtype is not None else None

    h_index, w_index = layout.index("H"), layout.index("W")
    in_h, in_w = in_shape[h_index], in_shape[w_index]
    if (
        coordinate_transformation_mode == "asymmetric"
        and rounding_method == ""
        and cubic_alpha == -0.5
        and cubic_exclude == 0
    ):
        scale_h, scale_w = out_h / in_h, out_w / in_w
        X = px.ops.upsampling2d(
            op_name,
            in_xlayers,
            scale_h=scale_h,
            scale_w=scale_w,
            data_layout=layout,
            method=method,
            relay_id=[hash(expr)],
        )
    else:
        # AnyOp
        out_shape = in_shape.tolist()
        out_shape[h_index] = out_h
        out_shape[w_index] = out_w
        X = px.ops.any_op(
            op_name, in_xlayers, any_shape=out_shape[:], relay_id=[hash(expr)]
        )

    logger.debug("-- outshape: {}".format(list(X.shapes)))
    return X


@register_relay_2_xlayer_converter_base("vision.yolo_reorg")
def yolo_reorg(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Conversion of Relay 'yolo_reorg' layer

    Relay
    -----
    Type: tvm.relay.vision.yolo_reorg
    Ref: https://docs.tvm.ai/langref/relay_op.html
    Parameters:
        - data (relay.Expr)
            The input data tensor.
        - stride (int)
            The stride value for reorganisation.
    """
    stride = int(expr.attrs.stride)
    X = px.ops.yolo_reorg(op_name, in_xlayers[0], stride, "NCHW", relay_id=[hash(expr)])
    logger.debug("-- outshape: {}".format(list(X.shapes)))
    return X


@register_relay_2_xlayer_converter_base("vision.get_valid_counts")
def vision_get_valid_counts(
    op_name: str, expr: Expr, in_xlayers: List[XLayer]
) -> XLayer:
    """
    TVM: Get valid count of bounding boxes given a score threshold. Also moves valid boxes
    to the top of input data.

    Relay
    -----
    Type: tvm.relay.strided_slice
    Ref: https://docs.tvm.ai/api/python/relay/vision.html
    Parameters:
        - data (relay.Expr)
            Input data. 3-D tensor with shape [batch_size, num_anchors, 6].
        - score_threshold (optional, float)
            Lower limit of score for valid bounding boxes.
        - id_index (optional, int)
            index of the class categories, -1 to disable.
        - score_index (optional, int)
            Index of the scores/confidence of boxes.
    """
    in_shape = list(in_xlayers[0].shapes[:])
    valid_count_shape = [1]
    out_tensor_shape = in_shape
    out_indices_shape = in_shape[:2]

    X = px.ops.any_op(
        op_name,
        in_xlayers,
        any_shape=[valid_count_shape, out_tensor_shape, out_indices_shape],
        relay_id=[hash(expr)],
    )
    return X


@register_relay_2_xlayer_converter_base("vision.non_max_suppression")
def vision_nms(op_name: str, expr: Expr, in_xlayers: List[XLayer]) -> XLayer:
    """
    Non-max suppression operation

    Relay
    -----
    Type: tvm.relay.vision.non_max_suppression
    Ref: https://docs.tvm.ai/api/python/relay/vision.html
    Parameters:
        - data (relay.Expr)
            3-D tensor with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5]. 
            The last dimension should be in format of [class_id, score, box_left, box_top, box_right,
            box_bottom] or [score, box_left, box_top, box_right, box_bottom]. It could be the second
            output out_tensor of get_valid_counts.
        - valid_count (relay.Expr)
            1-D tensor for valid number of boxes. It could be the output valid_count of get_valid_counts.
        - indices (relay.Expr)
            2-D tensor with shape [batch_size, num_anchors], represents the index of box in original data.
            It could be the third output out_indices of get_valid_counts. The values in the second
            dimension are like the output of arange(num_anchors) if get_valid_counts is not used before
            non_max_suppression.
        - max_output_size (int or relay.Expr, optional)
            Max number of output valid boxes for each instance. Return all valid boxes if the value of
            max_output_size is less than 0.
        - iou_threshold (float, optional)
            Non-maximum suppression threshold.
        - force_suppress (bool, optional)
            Suppress all detections regardless of class_id.
        - top_k (int, optional)
            Keep maximum top k detections before nms, -1 for no limit.
        - coord_start (int, optional)
            The starting index of the consecutive 4 coordinates.
        - score_index (int, optional)
            Index of the scores/confidence of boxes.
        - id_index (int, optional)
            index of the class categories, -1 to disable.
        - return_indices (bool, optional)
            Whether to return box indices in input data.
        - invalid_to_bottom (bool, optional)
            Whether to move all valid bounding boxes to the top.
    Returns
        - out
            return relay.Expr if return_indices is disabled, a 3-D tensor with shape
            [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5]. If return_indices is True,
            return relay.Tuple of two 2-D tensors, with shape [batch_size, num_anchors] and
            [batch_size, num_valid_anchors] respectively.
    """
    data_shape = list(in_xlayers[0].shapes[:])
    return_indices = bool(expr.attrs.return_indices)

    if not return_indices:
        newshape = data_shape[:]
    else:
        newshape = [[data_shape[0], data_shape[1]], [data_shape[0], -1]]

    X = px.ops.any_op(op_name, in_xlayers, any_shape=newshape, relay_id=[hash(expr)])
    return X
