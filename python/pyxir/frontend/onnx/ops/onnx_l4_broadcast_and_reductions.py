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
Module for transforming ONNX L4 operators to XLayer objects

L4: Broadcast and Reduction Operators


"""

import math
import logging
import numpy as np
import pyxir as px

from typing import Dict

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from pyxir.shapes.tools import get_numpy_broadcasted_shape
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper

logger = logging.getLogger('pyxir')


@register_onnx_2_xlayer_converter("ArgMax")
def argmax(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX Argmax to XLayer AnyOp conversion function """

    logger.info("ONNX Argmax -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else 0
    keepdims = bool(int(node_attrs['keepdims'])) if 'keepdims' in node_attrs \
        else True

    in_shape = iX.shapes.tolist()
    if keepdims:
        shape = [i if idx != axis else 1 for idx, i in enumerate(in_shape)]
    else:
        shape = [i for idx, i in enumerate(in_shape) if idx != axis]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ArgMin")
def argmin(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX ArgMin to XLayer AnyOp conversion function """

    logger.info("ONNX ArgMin -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else 0
    keepdims = bool(int(node_attrs['keepdims'])) if 'keepdims' in node_attrs \
        else True

    in_shape = iX.shapes.tolist()
    if keepdims:
        shape = [i if idx != axis else 1 for idx, i in enumerate(in_shape)]
    else:
        shape = [i for idx, i in enumerate(in_shape) if idx != axis]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Compress")
def compress(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    """ ONNX Compress to XLayer AnyOp conversion function
    Based on
    https://numpy.org/doc/stable/reference/generated/numpy.compress.html
    """

    logger.info("ONNX Compress -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    cX = xmap[bottoms[1]]
    d = len(iX.shapes)

    if 'Constant' not in cX.type:
        raise NotImplementedError("ONNX Compress operation with dynamic"
                                  " condition tensor unsupported")
    c_tensor = cX.data[0]

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else None
    if axis < 0:
        axis = d + axis

    in_shape = iX.shapes.tolist()
    if axis == 0:
        shape = [np.sum(c_tensor)]
    else:
        shape = [i if idx != axis else int(np.sum(c_tensor))
                 for idx, i in enumerate(in_shape)]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("DepthToSpace")
def depth_to_space(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX DepthToSpace to XLayer AnyOp conversion function """

    logger.info("ONNX DepthToSpace -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    in_b, in_c, in_h, in_w = iX.shapes.tolist()

    blocksize = int(node_attrs['blocksize'])
    shape = [in_b, in_c // (blocksize ** 2),
             in_h * blocksize, in_w * blocksize]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Equal")
def equal(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX Equal to XLayer AnyOp conversion function """

    logger.info("ONNX Equal -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Expand")
def expand(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX Expand to XLayer AnyOp conversion function """

    logger.info("ONNX Expand -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]
    expand_shape = [int(i) for i in bX.data[0]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        expand_shape)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Gather")
def gather(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX Gather to XLayer Take conversion function """

    logger.info("ONNX Gather -> XLayer Take")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    indices = xmap[bottoms[1]]

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else 0
    if axis < 0:
        axis = axis + d

    X = px.ops.take(
        op_name=px.stringify(name),
        in_xlayers=[iX, indices],
        axis=axis,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("GatherElements")
def gather_elements(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    """ ONNX GatherElements to XLayer AnyOp conversion function """

    logger.info("ONNX GatherElements -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    indices = xmap[bottoms[1]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX, indices],
        any_shape=indices.shapes.tolist(),
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("GatherND")
def gather_nd(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]):
    """ ONNX GatherND to XLayer AnyOp conversion function """

    logger.info("ONNX GatherND -> XLayer AnyOp")

    raise NotImplementedError("ONNX GatherND operator translation not"
                              " supported")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    indices = xmap[bottoms[1]].data[0]

    batch_dims = node_attrs['batch_dims'] if 'batch_dims' in node_attrs \
        else 0

    in_shape = iX.shapes.tolist()
    if indices[-1] == (d - batch_dims):
        out_shape = in_shape[:batch_dims] + [len(indices)]
    out_shape = []
    for elem in indices:
        d = batch_dims + elem.flatten().shape[0]
        out_shape.extend(in_shape[d:])

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Greater")
def greater(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    """ ONNX Greater to XLayer AnyOp conversion function """

    logger.info("ONNX Greater -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("GreaterOrEqual")
def greater_or_equal(node: NodeWrapper,
                     params: Dict[str, np.ndarray],
                     xmap: Dict[str, XLayer]):
    """ ONNX GreaterOrEqual to XLayer AnyOp conversion function """

    logger.info("ONNX GreaterOrEqual -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Less")
def less(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX Less to XLayer AnyOp conversion function """

    logger.info("ONNX Less -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("LessOrEqual")
def lessorequal(node: NodeWrapper,
                params: Dict[str, np.ndarray],
                xmap: Dict[str, XLayer]):
    """ ONNX LessOrEqual to XLayer AnyOp conversion function """

    logger.info("ONNX LessOrEqual -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Max")
def max(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX Max to XLayer AnyOp conversion function """

    logger.info("ONNX Max -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    aX = xmap[bottoms[0]]
    in_shape = aX.shapes.tolist()

    in_xlayers = [aX]

    for i in range(1, len(bottoms)):
        bX = xmap[bottoms[i]]
        in_shape = get_numpy_broadcasted_shape(in_shape,
                                               bX.shapes.tolist())
        in_xlayers.append(bX)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=in_xlayers,
        any_shape=in_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Mean")
def mean(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX Mean to XLayer AnyOp conversion function """

    logger.info("ONNX Mean -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    aX = xmap[bottoms[0]]
    in_shape = aX.shapes.tolist()

    in_xlayers = [aX]

    for i in range(1, len(bottoms)):
        bX = xmap[bottoms[i]]
        in_shape = get_numpy_broadcasted_shape(in_shape,
                                               bX.shapes.tolist())
        in_xlayers.append(bX)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=in_xlayers,
        any_shape=in_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Min")
def min(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX Min to XLayer AnyOp conversion function """

    logger.info("ONNX Min -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    aX = xmap[bottoms[0]]
    in_shape = aX.shapes.tolist()

    in_xlayers = [aX]

    for i in range(1, len(bottoms)):
        bX = xmap[bottoms[i]]
        in_shape = get_numpy_broadcasted_shape(in_shape,
                                               bX.shapes.tolist())
        in_xlayers.append(bX)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=in_xlayers,
        any_shape=in_shape,
        onnx_id=name
    )

    return [X]


def generic_reduce(op_type: str,
                   node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX Reduce to XLayer AnyOp conversion function """

    logger.info("ONNX {} -> XLayer AnyOp".format(op_type))

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    in_shape = iX.shapes.tolist()

    axes = [int(i) if i > 0 else int(d + i) for i in node_attrs['axes']]
    keepdims = bool(node_attrs['keepdims']) if 'keepdims' in node_attrs \
        else True

    out_shape = [(i if idx not in axes else 1)
                 for idx, i in enumerate(in_shape)
                 if (keepdims or i not in axes)]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ReduceL1")
def reducel1(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    """ ONNX ReduceL1 to XLayer AnyOp conversion function """
    return generic_reduce("ReduceL1", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceL2")
def reducel2(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    """ ONNX ReduceL2 to XLayer AnyOp conversion function """
    return generic_reduce("ReduceL2", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceLogSum")
def reducelogsum(node: NodeWrapper,
                 params: Dict[str, np.ndarray],
                 xmap: Dict[str, XLayer]):
    """ ONNX ReduceLogSum to XLayer AnyOp conversion function """
    return generic_reduce("ReduceLogSum", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceLogSumExp")
def reducelogsumexp(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    """ ONNX ReduceLogSumExp to XLayer AnyOp conversion function """
    return generic_reduce("ReduceLogSumExp", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceMax")
def reducemax(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]):
    """ ONNX ReduceMax to XLayer AnyOp conversion function """
    return generic_reduce("ReduceMax", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceMean")
def reducemean(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]):
    """ ONNX ReduceMean to XLayer AnyOp conversion function """
    return generic_reduce("ReduceMean", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceMin")
def reducemin(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]):
    """ ONNX ReduceMin to XLayer AnyOp conversion function """
    return generic_reduce("ReduceMin", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceProd")
def reduceprod(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]):
    """ ONNX ReduceProd to XLayer AnyOp conversion function """
    return generic_reduce("ReduceProd", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceSum")
def reducesum(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]):
    """ ONNX ReduceSum to XLayer AnyOp conversion function """
    return generic_reduce("ReduceSum", node, params, xmap)


@register_onnx_2_xlayer_converter("ReduceSumSquare")
def reducesumsquare(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    """ ONNX ReduceSumSquare to XLayer AnyOp conversion function """
    return generic_reduce("ReduceSumSquare", node, params, xmap)


@register_onnx_2_xlayer_converter("Resize")
def resize(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX Resize to XLayer AnyOp conversion function """

    logger.info("ONNX Resize -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    in_shape = iX.shapes.tolist()

    if len(bottoms) == 4:
        out_shape = [int(i) for i in list(xmap[bottoms[3]].data[0])] \
            if len(bottoms) == 4 else None
    else:
        roi = [float(i) for i in list(xmap[bottoms[1]].data[0])]
        scales = [float(i) for i in list(xmap[bottoms[2]].data[0])]

        h = len(in_shape)

        out_shape = [math.floor(in_dim * (roi[idx + h] - roi[idx])
                                * scales[idx])
                     for idx, in_dim in enumerate(in_shape)]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("RoiAlign")
def roi_align(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]):
    """ ONNX RoiAlign to XLayer AnyOp conversion function """

    logger.info("ONNX RoiAlign -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    _, in_c, _, _ = iX.shapes.tolist()
    in_shape = iX.shapes.tolist()

    rois = xmap[bottoms[1]]
    num_rois, _ = rois.shapes.tolist()

    out_h = node_attrs['output_height'] if 'output_height' in node_attrs \
        else 1
    out_w = node_attrs['output_width'] if 'output_width' in node_attrs \
        else 1

    out_shape = [num_rois, in_c, out_h, out_w]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Scatter")
def scatter(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    """ ONNX Scatter to XLayer AnyOp conversion function """

    logger.info("ONNX Scatter -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=iX.shapes.tolist(),
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ScatterElements")
def scatter_elements(node: NodeWrapper,
                     params: Dict[str, np.ndarray],
                     xmap: Dict[str, XLayer]):
    """ ONNX ScatterElements to XLayer AnyOp conversion function """

    logger.info("ONNX ScatterElements -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=iX.shapes.tolist(),
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ScatterND")
def scatter_nd(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]):
    """ ONNX ScatterND to XLayer AnyOp conversion function """

    logger.info("ONNX ScatterND -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=iX.shapes.tolist(),
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("SequenceAt")
def sequence_at(node: NodeWrapper,
                params: Dict[str, np.ndarray],
                xmap: Dict[str, XLayer]):
    """ ONNX SequenceAt to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX SequenceAt operator unsupported in Pyxir")


@register_onnx_2_xlayer_converter("SequenceConstruct")
def sequence_construct(node: NodeWrapper,
                       params: Dict[str, np.ndarray],
                       xmap: Dict[str, XLayer]):
    """ ONNX SequenceConstruct to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX SequenceAt operator unsupported in Pyxir")


@register_onnx_2_xlayer_converter("SequenceEmpty")
def sequence_empty(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX SequenceEmpty to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX SequenceEmpty operator unsupported in"
                              " Pyxir")


@register_onnx_2_xlayer_converter("SequenceErase")
def sequence_erase(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX SequenceErase to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX SequenceErase operator unsupported in"
                              " Pyxir")


@register_onnx_2_xlayer_converter("SequenceLength")
def sequence_length(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    """ ONNX SequenceErase to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX SequenceErase operator unsupported in"
                              " Pyxir")


@register_onnx_2_xlayer_converter("Slice")
def slice_op(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    """ ONNX Slice to XLayer AnyOp conversion function """

    logger.info("ONNX Slice -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    in_shape = iX.shapes.tolist()

    starts = [int(i) for i in list(xmap[bottoms[1]].data[0])]
    ends = [int(i) for i in list(xmap[bottoms[2]].data[0])]
    axes = [int(i) if i > 0 else (i + d)
            for i in list(xmap[bottoms[3]].data[0])]
    steps = [int(i) for i in list(xmap[bottoms[4]].data[0])]
    axes_to_sizes = {axis: math.floor((ends[idx] - starts[idx]) / steps[idx])
                     for idx, axis in enumerate(axes)}

    out_shape = [i if idx not in axes else axes_to_sizes[idx]
                 for idx, i in enumerate(in_shape)]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("SpaceToDepth")
def space_to_depth(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX SpaceToDepth to XLayer AnyOp conversion function """

    logger.info("ONNX SpaceToDepth -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    in_b, in_c, in_h, in_w = iX.shapes.tolist()

    blocksize = int(node_attrs['blocksize'])
    shape = [in_b, in_c * blocksize * blocksize,
             in_h // blocksize, in_w // blocksize]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("SplitToSequence")
def split_to_sequence(node: NodeWrapper,
                      params: Dict[str, np.ndarray],
                      xmap: Dict[str, XLayer]):
    """ ONNX SplitToSequence to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX SplitToSequence operator unsupported in"
                              " Pyxir")


@register_onnx_2_xlayer_converter("TfldfVectorizer")
def tfldf_vectorizer(node: NodeWrapper,
                     params: Dict[str, np.ndarray],
                     xmap: Dict[str, XLayer]):
    """ ONNX TfldfVectorizer to XLayer AnyOp conversion function """
    raise NotImplementedError("ONNX TfldfVectorizer operator unsupported in"
                              " Pyxir")


@register_onnx_2_xlayer_converter("Where")
def where(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX Where to XLayer AnyOp conversion function """

    logger.info("ONNX Where -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 3

    aX = xmap[bottoms[1]]
    bX = xmap[bottoms[2]]

    shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                        bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=shape,
        onnx_id=name
    )

    return [X]
