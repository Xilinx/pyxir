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
Module for transforming ONNX L1 operators to XLayer objects

L1: Basic NN operators that enable fully connected multi-layer perceptron
"""

import logging
import numpy as np
import pyxir as px

from typing import Dict, List

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.graph.layer import XLayer
from pyxir.shapes import get_numpy_broadcasted_shape
from ..onnx_2_xlayer_registry import register_onnx_2_xlayer_converter
from ..onnx_tools import NodeWrapper
from .tools import eltwise_any_op

logger = logging.getLogger("pyxir")


@register_onnx_2_xlayer_converter("Add")
def add(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ ONNX Add to XLayer Add conversion function """

    logger.info("ONNX Add -> XLayer Add")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    assert len(bottoms) == 2
    node_attrs = node.get_attributes()

    aX = xmap[bottoms[0]]  # NCHW
    bX = xmap[bottoms[1]]  # NCHW

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    def get_tensor_constant_add(tX, bias_X):

        if len(bias_X.shapes) == 1:
            X = xlf.get_xop_factory_func('BiasAdd')(
                op_name=px.stringify(name),
                axis=1,
                input_layer=tX,
                bias_layer=bias_X,
                onnx_id=name
            )
            Xs_tmp = [X]
        else:
            X = px.ops.add(
                op_name=px.stringify(name),
                in_xlayers=[tX, bias_X],
                onnx_id=name
            )
            Xs_tmp = [bias_X, X]

        return Xs_tmp

    if 'Constant' in aX.type and 'Constant' in bX.type:
        data = np.add(aX.data[0], bX.data[0])
        X = xlf.get_xop_factory_func('Constant')(
            op_name=px.stringify(name),
            value=data,
            onnx_id=name
        )
        Xs = [X]
    elif 'Constant' in aX.type and 'Constant' not in bX.type:
        Xs = get_tensor_constant_add(bX, aX)
    elif 'Constant' in bX.type and 'Constant' not in aX.type:
        Xs = get_tensor_constant_add(aX, bX)
    else:
        X = xlf.get_xop_factory_func('Eltwise')(
            op_name=px.stringify(name),
            lhs_layer=aX,
            rhs_layer=bX,
            vai_quant=vai_quant,
            vai_quant_in=vai_quant_in,
            vai_quant_out=vai_quant_out,
            onnx_id=name
        )
        Xs = [X]

    return Xs


@register_onnx_2_xlayer_converter("BatchNormalization")
def batchnorm(node: NodeWrapper,
              params: Dict[str, np.ndarray],
              xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX BatchNormalization to XLayer BatchNorm conversion function"""
    logger.info("ONNX BatchNorm -> XLayer BatchNorm")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    _, in_c, in_h, in_w = iX.shapes

    scale_name = bottoms[1]
    sX = xmap[scale_name]  # C
    assert sX.shapes[0] == in_c

    bias_name = bottoms[2]
    bX = xmap[bias_name]  # C
    assert bX.shapes[0] == in_c

    mean_name = bottoms[3]
    mX = xmap[mean_name]  # C
    assert mX.shapes[0] == in_c

    var_name = bottoms[4]
    vX = xmap[var_name]  # C
    assert vX.shapes[0] == in_c

    epsilon = node_attrs['epsilon'] if 'epsilon' in node_attrs\
        else 1e-05
    # Ignore momentum
    momentum = node_attrs['momentum'] if 'momentum' in node_attrs\
        else 0.9

    X = xlf.get_xop_factory_func('BatchNorm')(
        op_name=px.stringify(name),
        axis=1,
        epsilon=epsilon,
        input_layer=iX,
        mean_layer=mX,
        variance_layer=vX,
        gamma_layer=sX,
        beta_layer=bX,
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Concat")
def concat(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Concat to XLayer Concat conversion function"""
    logger.info("ONNX Concat -> XLayer Concat")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    Xs = [xmap[b] for b in bottoms]
    d = len(Xs[0].shapes)

    axis = int(node_attrs['axis'])
    if axis < 0:
        axis = d + axis

    X = px.ops.concat(
        op_name=px.stringify(name),
        input_layers=Xs,
        axis=axis,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("ConcatFromSequence")
def concat_from_sequence(node: NodeWrapper,
                         params: Dict[str, np.ndarray],
                         xmap: Dict[str, XLayer]):
    raise NotImplementedError("")


@register_onnx_2_xlayer_converter("Div")
def div(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Div to XLayer Divide conversion function"""
    logger.info("ONNX And -> XLayer Divide")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    X = px.ops.divide(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Dropout")
def dropout(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Dropout to XLayer Dropout conversion function"""
    logger.info("ONNX Dropout -> XLayer Dropout")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]

    ratio = float(node_attrs['ratio']) if 'ratio' in node_attrs else 0.5

    X = px.ops.dropout(
        op_name=px.stringify(name),
        input_layer=iX,
        rate=ratio,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Exp")
def exp(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Exp to XLayer Exp conversion function"""

    logger.info("ONNX Exp -> XLayer Exp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]  # NCHW

    X = px.ops.exp(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Gemm")
def gemm(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]) -> List[XLayer]:
    """
    ONNX Gemm to XLayer Dense (+ Scale) (+ BiasAdd) conversion function

    Compute Y = alpha * A' * B' + beta * C
    See https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
    """
    logger.info("ONNX Gemm-> XLayer Dense (+ Scale) (+ BiasAdd)")

    assert len(node.get_outputs()) == 1
    assert len(node.get_inputs()) in [2, 3]
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NC or CN
    _, in_c = iX.shapes

    W_name = bottoms[1]
    wX = xmap[W_name]
    assert len(wX.shapes) == 2

    B_name = bottoms[2] if len(bottoms) == 3 else None
    bX = xmap[B_name] if len(bottoms) == 3 else None

    alpha = node_attrs['alpha'] if 'alpha' in node_attrs else 1.0
    beta = node_attrs['beta'] if 'beta' in node_attrs else 1.0
    trans_A = node_attrs['transA'] > 0 if 'transA' in node_attrs else False
    trans_B = node_attrs['transB'] > 0 if 'transB' in node_attrs else False

    if alpha != 1.0:
        raise NotImplementedError("Alpha != 1.0 not supported in ONNX Gemm to"
                                  " XLayer Dense conversion")
    if beta != 1.0:
        raise NotImplementedError("Beta != 1.0 not supported in ONNX Gemm to"
                                  " XLayer Dense conversion")

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in'] \
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out'] \
        if 'vai_quant_out' in node_attrs else []
    vai_quant_weights = node_attrs['vai_quant_weights'] \
        if 'vai_quant_weights' in node_attrs else []
    vai_quant_biases = node_attrs['vai_quant_biases'] \
        if 'vai_quant_biases' in node_attrs else []
    vai_quant = node_attrs['vai_quant'] \
        if 'vai_quant' in node_attrs else []
    vai_quant_dense = [a for a in vai_quant if str(a) != 'vai_quant_biases']
    vai_quant_bias_add = [a for a in vai_quant if str(a) == 'vai_quant_biases']

    Xs = []

    if trans_A:
        # iX is in CN -> Transform to NC
        iX = xlf.get_xop_factory_func('Transpose')(
            op_name=iX.name + '_transpose',
            axes=[1, 0],
            input_layer=iX,
            onnx_id=name
        )
        Xs.append(iX)

    if not trans_B:
        # iX is in IO -> Transform to OI
        wX = xlf.get_xop_factory_func('Transpose')(
            op_name=W_name + '_transpose',
            axes=[1, 0],
            input_layer=wX,
            onnx_id=name
        )
        Xs.append(wX)

    units = wX.shapes[0]

    dense_name = name if B_name is None else name + '_Dense'
    X = xlf.get_xop_factory_func('Dense')(
        op_name=px.stringify(dense_name),
        units=units,
        input_layer=iX,
        weights_layer=wX,
        vai_quant=vai_quant_dense,
        vai_quant_in=vai_quant_in,
        vai_quant_out=vai_quant_out,
        vai_quant_weights=vai_quant_weights,
        onnx_id=name
    )
    Xs.append(X)

    if B_name is not None:

        bias_add_X = xlf.get_xop_factory_func('BiasAdd')(
            op_name=px.stringify(name),
            axis=1,
            input_layer=X,
            bias_layer=bX,
            vai_quant=vai_quant_bias_add,
            vai_quant_biases=vai_quant_biases,
            onnx_id=name
        )

        Xs.append(bias_add_X)

    return Xs


@register_onnx_2_xlayer_converter("InstanceNormalization")
def instancenormalization(node: NodeWrapper,
                          params: Dict[str, np.ndarray],
                          xmap: Dict[str, XLayer]) -> List[XLayer]:
    return eltwise_any_op("InstanceNormalization", node, params, xmap)


@register_onnx_2_xlayer_converter("Log")
def log(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]):
    """ONNX Log to XLayer Log conversion function"""

    logger.info("ONNX Log -> XLayer Log")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]  # NCHW

    X = px.ops.log(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("LogSoftmax")
def logsoftmax(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]):
    return eltwise_any_op("LogSoftmax", node, params, xmap)


@register_onnx_2_xlayer_converter("MatMul")
def matmul(node: NodeWrapper,
           params: Dict[str, np.ndarray],
           xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX MatMul to XLayer Dense/AnyOp conversion function"""
    logger.info("ONNX MatMul -> XLayer Dense/AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    units = bX.shapes[1]

    if 'Constant' in bX.type:
        X = px.ops.dense(
            op_name=px.stringify(name),
            input_layer=iX,
            weights_layer=bX,
            units=units,
            kernel_layout='IO',
            onnx_id=name
        )
    else:
        X = px.ops.any_op(
            op_name=px.stringify(name),
            in_xlayers=[iX],
            any_shape=[iX.shapes[0], units],
            onnx_id=name
        )

    return [X]


@register_onnx_2_xlayer_converter("MatMulInteger")
def matmul_integer(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX MatMulInteger to XLayer Dense/AnyOp conversion function"""
    return matmul(node, params, xmap)


@register_onnx_2_xlayer_converter("Mod")
def mod(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Mod to XLayer AnyOp conversion function"""
    logger.info("ONNX Mod -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) == 2

    aX = xmap[bottoms[0]]
    bX = xmap[bottoms[1]]

    out_shape = get_numpy_broadcasted_shape(aX.shapes.tolist(),
                                            bX.shapes.tolist())

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        any_shape=out_shape,
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Mul")
def mul(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: XLayer) -> List[XLayer]:
    """ONNX Mul to XLayer conversion function"""

    logger.info("ONNX Mul -> XLayer Multiply")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    assert len(bottoms) == 2
    node_attrs = node.get_attributes()

    aX = xmap[bottoms[0]]  # NCHW
    bX = xmap[bottoms[1]]  # NCHW

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    if 'Constant' in aX.type and 'Constant' in bX.type:
        data = np.multiply(aX.data[0], bX.data[0])
        X = xlf.get_xop_factory_func('Constant')(
            op_name=px.stringify(name),
            value=data,
            onnx_id=name
        )
        Xs = [X]
    elif 'Constant' in aX.type and 'Constant' not in bX.type:
        beta_X = px.ops.constant(
            op_name='beta-' + px.stringify(name),
            value=np.array(aX.data[0] * 0., dtype=np.float32),
            internal=True,
            onnx_id=name
        )

        X = px.ops.scale(
            op_name=px.stringify(name),
            input_layer=bX,
            gamma_layer=aX,
            beta_layer=beta_X,
            axis=-1,
            onnx_id=name)
        Xs = [beta_X, X]
    elif 'Constant' in bX.type and 'Constant' not in aX.type:
        beta_X = px.ops.constant(
            op_name='beta-' + px.stringify(name),
            value=np.array(bX.data[0] * 0., dtype=np.float32),
            internal=True,
            onnx_id=name
        )

        X = px.ops.scale(
            op_name=px.stringify(name),
            input_layer=aX,
            gamma_layer=bX,
            beta_layer=beta_X,
            axis=-1,
            onnx_id=name)
        Xs = [beta_X, X]
    else:
        X = px.ops.multiply(
            op_name=px.stringify(name),
            in_xlayers=[aX, bX],
            vai_quant=vai_quant,
            vai_quant_in=vai_quant_in,
            vai_quant_out=vai_quant_out,
            onnx_id=name
        )
        Xs = [X]

    return Xs


@register_onnx_2_xlayer_converter("QLinearMatMul")
def qlinear_matmul(node: NodeWrapper,
                   params: Dict[str, np.ndarray],
                   xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX QLinearMatMul to XLayer AnyOp conversion function"""
    logger.info("ONNX QLinearMatMul -> XLayer AnyOp")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]
    bX = xmap[bottoms[3]]

    units = bX.shapes[1]

    X = px.ops.any_op(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        any_shape=[iX.shapes[0], units],
        onnx_id=name
    )

    return [X]


@register_onnx_2_xlayer_converter("Reciprocal")
def reciprocal(node: NodeWrapper,
               params: Dict[str, np.ndarray],
               xmap: Dict[str, XLayer]) -> List[XLayer]:
    return eltwise_any_op("Reciprocal", node, params, xmap)


@register_onnx_2_xlayer_converter("Relu")
def relu(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Relu to XLayer ReLU conversion function"""

    logger.info("ONNX Relu -> XLayer Relu")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]  # NCHW
    X = px.ops.relu(px.stringify(name), [iX], onnx_id=name)
    return [X]


@register_onnx_2_xlayer_converter("Sigmoid")
def sigmoid(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Sigmoid to XLayer Sigmoid conversion function"""
    logger.info("ONNX Sigmoid -> XLayer Sigmoid")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]  # NCHW

    X = px.ops.sigmoid(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Softmax")
def softmax(node: NodeWrapper,
            params: Dict[str, np.ndarray],
            xmap: Dict[str, XLayer])  -> List[XLayer]:
    """ONNX Softmax to XLayer Softmax conversion function"""
    logger.info("ONNX Softmax -> XLayer Softmax")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()

    iX = xmap[bottoms[0]]  # NCHW
    d = len(iX.shapes)

    axis = int(node_attrs['axis']) if 'axis' in node_attrs else 1
    if axis < 0:
        axis += d

    X = px.ops.softmax(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        axis=axis,
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Sqrt")
def sqrt(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Sqrt to XLayer Sqrt conversion function"""
    logger.info("ONNX Sqrt -> XLayer Sqrt")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]  # NCHW

    X = px.ops.sqrt(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        onnx_id=name)

    return [X]


@register_onnx_2_xlayer_converter("Sub")
def sub(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Sub to XLayer Sub conversion function"""
    logger.info("ONNX Sub -> XLayer Sub")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    assert len(bottoms) == 2
    node_attrs = node.get_attributes()

    aX = xmap[bottoms[0]]  # NCHW
    bX = xmap[bottoms[1]]  # NCHW

    # Quant_info (optional)
    vai_quant_in = node_attrs['vai_quant_in']\
        if 'vai_quant_in' in node_attrs else []
    vai_quant_out = node_attrs['vai_quant_out']\
        if 'vai_quant_out' in node_attrs else []
    vai_quant = node_attrs['vai_quant']\
        if 'vai_quant' in node_attrs else []

    if 'Constant' in aX.type and 'Constant' in bX.type:
        data = np.subtract(aX.data[0], bX.data[0])
        X = xlf.get_xop_factory_func('Constant')(
            op_name=px.stringify(name),
            value=data,
            onnx_id=name
        )
    else:
        X = px.ops.sub(
            op_name=px.stringify(name),
            in_xlayers=[aX, bX],
            vai_quant=vai_quant,
            vai_quant_in=vai_quant_in,
            vai_quant_out=vai_quant_out,
            onnx_id=name
        )

    return [X]


@register_onnx_2_xlayer_converter("Sum")
def sum(node: NodeWrapper,
        params: Dict[str, np.ndarray],
        xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Sum to XLayer Add conversion function"""
    logger.info("ONNX Sum -> XLayer Add")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()
    node_attrs = node.get_attributes()
    assert len(bottoms) >= 2

    aX = xmap[bottoms[0]]  # NCHW
    bX = xmap[bottoms[1]]  # NCHW

    Xs = []
    X = px.ops.add(
        op_name=px.stringify(name),
        in_xlayers=[aX, bX],
        onnx_id=name
    )
    Xs.append(X)

    for i in range(2, len(bottoms)):
        X = px.ops.add(
            op_name=px.stringify(name) + str(i),
            in_xlayers=[Xs[i - 2], xmap[bottoms[i]]],
            onnx_id=name
        )
        Xs.append(X)

    return Xs


@register_onnx_2_xlayer_converter("Tanh")
def tanh(node: NodeWrapper,
         params: Dict[str, np.ndarray],
         xmap: Dict[str, XLayer]) -> List[XLayer]:
    """ONNX Tanh to XLayer Tanh conversion function"""
    logger.info("ONNX Tanh -> XLayer Tanh")

    assert len(node.get_outputs()) == 1
    name = node.get_outputs()[0]
    bottoms = node.get_inputs()

    iX = xmap[bottoms[0]]  # NCHW

    X = px.ops.tanh(
        op_name=px.stringify(name),
        in_xlayers=[iX],
        onnx_id=name)

    return [X]
