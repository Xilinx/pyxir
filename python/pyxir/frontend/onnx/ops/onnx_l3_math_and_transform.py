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
Module for transforming ONNX L3 operators to XLayer objects

L3: Additional math and transform operators


"""

import logging
import numpy as np
import pyxir as px

from typing import Dict

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from pyxir.shapes.tools import get_numpy_broadcasted_shape
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper, get_onnx_elem_type_2_dtype
from .tools import eltwise_any_op

logger = logging.getLogger('pyxir')


@register_onnx_2_xlayer_converter("Abs")
def abs_op(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    return eltwise_any_op("Abs", node, params, xmap)


@register_onnx_2_xlayer_converter("Acos")
def acos(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Acos", node, params, xmap)


@register_onnx_2_xlayer_converter("Acosh")
def acosh(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("Acosh", node, params, xmap)


@register_onnx_2_xlayer_converter("And")
def and_op(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX And to XLayer AnyOp conversion function """

    logger.info("ONNX And -> XLayer AnyOp")

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


@register_onnx_2_xlayer_converter("Asin")
def asin(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Asin", node, params, xmap)


@register_onnx_2_xlayer_converter("Asinh")
def asinh(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("Asinh", node, params, xmap)


@register_onnx_2_xlayer_converter("Atan")
def atan(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Atan", node, params, xmap)


@register_onnx_2_xlayer_converter("Atanh")
def atanh(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("Atanh", node, params, xmap)


@register_onnx_2_xlayer_converter("BitShift")
def bitshift(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    return eltwise_any_op("BitShift", node, params, xmap)


@register_onnx_2_xlayer_converter("Cast")
def cast(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX Cast to XLayer Cast conversion function """

    logger.info("ONNX Cast -> XLayer Cast")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    to = int(node_attrs['to'])
    dtype = get_onnx_elem_type_2_dtype()[to]

    X = px.ops.cast(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        dtype=dtype,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Ceil")
def ceil(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Ceil", node, params, xmap)


@register_onnx_2_xlayer_converter("Celu")
def celu(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Celu", node, params, xmap)


@register_onnx_2_xlayer_converter("Clip")
def clip(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Clip", node, params, xmap)


@register_onnx_2_xlayer_converter("Cos")
def cos(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    return eltwise_any_op("Cos", node, params, xmap)


@register_onnx_2_xlayer_converter("Cosh")
def cosh(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Cosh", node, params, xmap)


@register_onnx_2_xlayer_converter("CumSum")
def cumsum(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    return eltwise_any_op("CumSum", node, params, xmap)


@register_onnx_2_xlayer_converter("Det")
def det(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX Det to XLayer AnyOp conversion function

    Input tensor shape: (*, M, M)
    Output tensor shape: (*)
    """

    logger.info("ONNX Det -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=[iX.shapes.tolist()[0]],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Einsum")
def einsum(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    """ ONNX Einsum to XLayer AnyOp conversion function """

    logger.info("ONNX Einsum -> XLayer AnyOp")

    raise NotImplementedError("Einsum operator conversion not supported")


@register_onnx_2_xlayer_converter("Elu")
def elu(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    return eltwise_any_op("Elu", node, params, xmap)


@register_onnx_2_xlayer_converter("Erf")
def erf(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    return eltwise_any_op("Erf", node, params, xmap)


@register_onnx_2_xlayer_converter("EyeLike")
def eyelike(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    return eltwise_any_op("EyeLike", node, params, xmap)


@register_onnx_2_xlayer_converter("Floor")
def floor(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("Floor", node, params, xmap)


@register_onnx_2_xlayer_converter("HardSigmoid")
def hardsigmoid(node: NodeWrapper,
                params: Dict[str, np.ndarray],
                xmap: Dict[str, XLayer]):
    return eltwise_any_op("HardSigmoid", node, params, xmap)


@register_onnx_2_xlayer_converter("Hardmax")
def hard_max(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    """ ONNX Hardmax to XLayer AnyOp conversion function

    Input tensor shape: N dims
    Output tensor shape: 2D
    """

    logger.info("ONNX Hardmax -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else 1
    if axis < 0:
        axis = d + axis

    in_shape = iX.shapes.tolist()
    dim_0 = int(np.prod(in_shape[:axis]))
    dim_1 = int(np.prod(in_shape[axis:]))

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=[dim_0, dim_1],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("IsInf")
def isinf(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("IsInf", node, params, xmap)


@register_onnx_2_xlayer_converter("IsNaN")
def isnan(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("IsNaN", node, params, xmap)


@register_onnx_2_xlayer_converter("LeakyRelu")
def leaky_relu(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]):
    """ ONNX LeakyRelu to XLayer LeakyRelu conversion function """

    logger.info("ONNX LeakyRelu -> XLayer LeakyRelu")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    alpha = float(node_attrs['alpha']) if 'alpha' in node_attrs else 0.01

    X = px.ops.leaky_relu(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        alpha=alpha,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("LpNormalization")
def lpnormalization(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    return eltwise_any_op("LpNormalization", node, params, xmap)


@register_onnx_2_xlayer_converter("MeanVarianceNormalization")
def meanvariancenormalization(node: NodeWrapper,
                              params: Dict[str, np.ndarray],
                              xmap: Dict[str, XLayer]):
    return eltwise_any_op("MeanVarianceNormalization", node, params, xmap)


@register_onnx_2_xlayer_converter("Multinomial")
def multinomial(node: NodeWrapper,
                params: Dict[str, np.ndarray],
                xmap: Dict[str, XLayer]):
    """ ONNX Multinomial to XLayer AnyOp conversion function """

    logger.info("ONNX Multinomial -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    batch_size, class_size = iX.shapes.tolist()

    sample_size = int(node_attrs['sample_size']) \
        if 'sample_size' in node_attrs else 1

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=[batch_size, sample_size],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Neg")
def neg(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    return eltwise_any_op("Neg", node, params, xmap)


@register_onnx_2_xlayer_converter("NegativeLogLikelihoodLoss")
def negative_log_likelihood_loss(node: NodeWrapper,
                                 params: Dict[str, np.ndarray],
                                 xmap: Dict[str, XLayer]):
    """ ONNX NegativeLogLikelihoodLoss to XLayer AnyOp conversion function """

    logger.info("ONNX NegativeLogLikelihoodLoss -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    out_shape = xmap[bottoms[1]].shapes.tolist()

    reduction = str(node_attrs['reduction']) \
        if 'reduction' in node_attrs else 'mean'
    if reduction != 'none':
        out_shape = [1]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("NonZero")
def nonzero(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    return eltwise_any_op("NonZero", node, params, xmap)


@register_onnx_2_xlayer_converter("OneHot")
def one_hot(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    """ ONNX OneHot to XLayer AnyOp conversion function """

    logger.info("ONNX OneHot -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)
    depth = int(xmap[bottoms[1]].data[0])

    axis = str(node_attrs['axis']) if 'axis' in node_attrs else -1
    if axis < 0:
        axis += d

    out_shape = iX.shapes.tolist()
    out_shape.insert(axis + 1, depth)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Or")
def or_op(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX Or to XLayer AnyOp conversion function """

    logger.info("ONNX Or -> XLayer AnyOp")

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


@register_onnx_2_xlayer_converter("PRelu")
def prelu(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX PRelu to XLayer PRelu conversion function """

    logger.info("ONNX LeakyRelu -> XLayer LeakyRelu")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    alpha = float(node_attrs['slope'])

    X = px.ops.leaky_relu(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        alpha=alpha,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Pow")
def pow(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX Pow to XLayer AnyOp conversion function """

    logger.info("ONNX Pow -> XLayer AnyOp")

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


@register_onnx_2_xlayer_converter("RandomNormal")
def random_normal(node: NodeWrapper,
                  params: Dict[str, np.ndarray],
                  xmap: Dict[str, XLayer]):
    """ ONNX RandomNormal to XLayer AnyOp conversion function """

    logger.info("ONNX RandomNormal -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    shape = [int(i) for i in node_attrs['shape']]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("RandomNormalLike")
def randomnormallike(node: NodeWrapper,
                     params: Dict[str, np.ndarray],
                     xmap: Dict[str, XLayer]):
    return eltwise_any_op("RandomNormalLike", node, params, xmap)


@register_onnx_2_xlayer_converter("RandomUniform")
def random_uniform(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]):
    """ ONNX RandomUniform to XLayer AnyOp conversion function """

    logger.info("ONNX RandomUniform -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    shape = [int(i) for i in node_attrs['shape']]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[],
        any_shape=shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("RandomUniformLike")
def randomuniformlike(node: NodeWrapper,
                      params: Dict[str, np.ndarray],
                      xmap: Dict[str, XLayer]):
    return eltwise_any_op("RandomUniformLike", node, params, xmap)


@register_onnx_2_xlayer_converter("Reshape")
def reshape(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    """ ONNX Reshape to XLayer Reshape conversion function """

    logger.info("ONNX Reshape -> XLayer Reshape")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    iX_size = iX.sizes

    onnx_newshape = [int(i) for i in list(xmap[bottoms[1]].data[0])]
    newshape = []
    minus_one_idx = None
    for idx, i in enumerate(onnx_newshape):
        if i == 0:
            newshape.append(iX.shapes[idx])
        elif i == -1:
            newshape.append(-1)
            minus_one_idx = idx
        else:
            newshape.append(i)
    if minus_one_idx is not None:
        newshape[minus_one_idx] = \
            int(iX_size[0] // abs(int(np.prod(newshape))))

    X = px.ops.reshape(
        op_name=px.stringify(name),
        input_layer=iX,
        newshape=newshape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ReverseSequence")
def reversesequence(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    return eltwise_any_op("ReverseSequence", node, params, xmap)


@register_onnx_2_xlayer_converter("Round")
def round(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    return eltwise_any_op("Round", node, params, xmap)


@register_onnx_2_xlayer_converter("Selu")
def selu(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Selu", node, params, xmap)


@register_onnx_2_xlayer_converter("Shrink")
def shrink(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]):
    return eltwise_any_op("Shrink", node, params, xmap)


@register_onnx_2_xlayer_converter("Sign")
def sign(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Sign", node, params, xmap)


@register_onnx_2_xlayer_converter("Sin")
def sin(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    return eltwise_any_op("Sin", node, params, xmap)


@register_onnx_2_xlayer_converter("Sinh")
def sinh(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    return eltwise_any_op("Sinh", node, params, xmap)


@register_onnx_2_xlayer_converter("SoftmaxCrossEntropyLoss")
def softmax_cross_entropy_loss(node: NodeWrapper,
                               params: Dict[str, np.ndarray],
                               xmap: Dict[str, XLayer]):
    """ ONNX SoftmaxCrossEntropyLoss to XLayer AnyOp conversion function """

    logger.info("ONNX SoftmaxCrossEntropyLoss -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    out_shape = xmap[bottoms[1]].shapes.tolist()

    reduction = str(node_attrs['reduction']) \
        if 'reduction' in node_attrs else 'mean'
    if reduction != 'none':
        out_shape = [1]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("SoftPlus")
def softplus(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    return eltwise_any_op("SoftPlus", node, params, xmap)


@register_onnx_2_xlayer_converter("SoftSign")
def softsign(node: NodeWrapper,
             params: Dict[str, np.ndarray],
             xmap: Dict[str, XLayer]):
    return eltwise_any_op("SoftSign", node, params, xmap)


@register_onnx_2_xlayer_converter("Split")
def split(node: NodeWrapper,
          params: Dict[str, np.ndarray],
          xmap: Dict[str, XLayer]):
    """ ONNX Split to XLayer Split conversion function """

    logger.info("ONNX Split -> XLayer Split")

    # TODO first name is used for split for now
    name = node.get_outputs()[0]
    nb_outputs = len(node.get_outputs())
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]
    d = len(iX.shapes)

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else 0
    if axis < 0:
        axis = d + axis

    split = [int(s) for s in node_attrs['split']] if 'split' in node_attrs \
        else [int(iX.shapes[axis] // nb_outputs)] * nb_outputs
    indices = [split[0]]
    for i in range(1, len(split) - 1):
        indices.append(indices[i - 1] + split[i])

    Xs = []
    X = px.ops.split(
        op_name='split-' + px.stringify(name),
        in_xlayers=[iX],
        axis=axis,
        indices=indices,
        onnx_id=name
    )
    Xs.append(X)

    for idx, tensor_name in enumerate(node.get_outputs()):
        tgi_X = px.ops.tuple_get_item(
            op_name=px.stringify(tensor_name),
            in_xlayers=[X],
            index=idx,
            onnx_id=tensor_name
        )
        Xs.append(tgi_X)

    return Xs


@register_onnx_2_xlayer_converter("Squeeze")
def squeeze(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]):
    """ ONNX Squeeze to XLayer Squeeze conversion function """

    logger.info("ONNX Squeeze -> XLayer Squeeze")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    d = len(iX.shapes)

    axes = [int(i) if i > 0 else (d + i) for i in node_attrs['axes']]

    X = px.ops.squeeze(
        op_name=px.stringify(name),
        input_layer=iX,
        axis=axes,
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Tan")
def tan(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    return eltwise_any_op("Tan", node, params, xmap)


@register_onnx_2_xlayer_converter("ThresholdedRelu")
def thresholdedrelu(node: NodeWrapper,
                    params: Dict[str, np.ndarray],
                    xmap: Dict[str, XLayer]):
    return eltwise_any_op("ThresholdedRelu", node, params, xmap)


@register_onnx_2_xlayer_converter("Tile")
def tile(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]):
    """ ONNX Tile to XLayer Tile conversion function """

    logger.info("ONNX Tile -> XLayer Tile")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    in_shape = iX.shapes.tolist()
    repeats = [int(i) for i in list(xmap[bottoms[1]].data[0])]

    out_shape = [in_shape[i] * repeats[i] for i in range(len(in_shape))]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Transpose")
def transpose(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]):
    """ ONNX Transpose to XLayer Transpose conversion function """

    logger.info("ONNX Transpose -> XLayer Transpose")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    axes = [int(e) for e in node_attrs['perm']] if 'perm' in node_attrs \
        else [i for i in range(len(iX.shapes) - 1, -1, -1)]

    X = px.ops.transpose(
        op_name=px.stringify(name),
        input_layer=iX,
        axes=axes,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Unsqueeze")
def un_squeeze(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]):
    """ ONNX UnSqueeze to XLayer AnyOp conversion function """

    logger.info("ONNX Unsqueeze -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    d = len(iX.shapes)

    axes = [int(i) if i > 0 else (d + i) for i in node_attrs['axes']]

    out_shape = iX.shapes.tolist()
    for axis in axes:
        out_shape.insert(axis, 1)

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=out_shape,
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Xor")
def xor(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX Xor to XLayer AnyOp conversion function """

    logger.info("ONNX Xor -> XLayer AnyOp")

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
