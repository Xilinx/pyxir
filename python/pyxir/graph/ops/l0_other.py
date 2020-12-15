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
Module for creating L0 XLayer objects

L0: Other, mostly input and graph utility operations like Tuple, TupleGetItem
"""


import math
import logging
import warnings
import numpy as np

from typing import Dict, List

import pyxir
from pyxir.shapes import TensorShape, TupleShape

from ..layer.xlayer import defaultXLayer, XLayer
from ..layer.xlayer_factory import xop_register_factory, xop_register

logger = logging.getLogger("pyxir")


############
# Constant #
############

@xop_register_factory('Constant')
def constant(op_name, value, **kwargs):
    # type: (str, numpy.ndarray) -> XLayer
    """
    Create a Constant parameters layer

    Arguments
    ---------
    op_name: str
        The name of this constant layer
    value: numpy.ndarray
        The value of this constant layer
    """
    if not isinstance(value, np.ndarray):
        value = np.array(value)

    dtype = str(value.dtype)

    attrs = kwargs
    attrs.update({'dtype': dtype})

    shape = TensorShape(list(value.shape))

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Constant'],
        shapes=shape,
        sizes=shape.get_size(),
        data=[value],
        layer=[op_name],
        tops=[],
        bottoms=[],
        attrs=attrs,
        targets=[]
    )

    return X


#########
# Input #
#########

@xop_register_factory('Input')
def input(op_name, shape, dtype='float32', **kwargs):
    # type: (str, List[int], str) -> XLayer
    """
    Create a Input parameters layer

    Arguments
    ---------
    op_name: str
        The name of this input layer
    shape: List[int]
        The input shape
    dtype: str (optional, default None)
        The input data type
    """

    shape[0] = -1
    shape = TensorShape(shape)

    attrs = kwargs
    attrs.update({
        'dtype': dtype
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Input'],
        shapes=shape,
        sizes=shape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=[],
        attrs=attrs,
        targets=[]
    )

    return X


@xop_register('Output')
def output(attrs, in_xlayers):
    # type: (str, List[XLayer]) -> XLayer
    """ Return Output registration information (shape) """

    assert len(in_xlayers) == 1

    shape = in_xlayers[0].shapes[:]

    return {'shape': shape}


############
# StrInput #
############

@xop_register_factory('StrInput')
def str_input(op_name, **kwargs):
    # type: (str) -> XLayer
    """
    Create a string input XLayer

    Arguments
    ---------
    op_name: str
        The name of this string input layer
    """

    attrs = kwargs

    X = defaultXLayer()
    X = X._replace(
        name=op_name,
        type=['StrInput'],
        shapes=TensorShape([-1]),
        sizes=[1],
        layer=[op_name],
        attrs=attrs,
    )

    return X


#########
# Tuple #
#########

@xop_register_factory('Tuple')
def tuple(op_name, input_layers, **kwargs):
    # type: (str, int, List[XLayer]) -> XLayer
    """
    Create an tuple XLayer for grouping a list of input layers

    Arguments
    ---------
    input_layers: List[XLayer]
        The input layers to be grouped in a tuple data structure
    """
    bottoms = [input_layer.name for input_layer in input_layers]
    shapes = TupleShape([TensorShape(il.shapes[:]) for il in input_layers])

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Tuple'],
        shapes=shapes,
        sizes=shapes.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=kwargs,
        targets=[]
    )

    return X


################
# TupleGetItem #
################

@xop_register('TupleGetItem')
def tuple_get_item(attrs, in_xlayers):
    # type: (str, List[XLayer]) -> XLayer
    """ Return TupleGetItem registration information (shape) """

    assert len(in_xlayers) == 1
    assert isinstance(in_xlayers[0].shapes, TupleShape)

    index = attrs['index']

    shape = in_xlayers[0].shapes[index][:]

    return {'shape': shape}


###########
# RelayOp #
###########

@xop_register('RelayOp')
def relay_op(attrs, in_xlayers):
    # type: (str, List[XLayer]) -> XLayer
    """ Return RelayOp registration information (shape) """

    relay_shape = attrs['relay_shape']
    if len(relay_shape) > 0 and isinstance(relay_shape[0], list):
        shape = TupleShape(attrs['relay_shape'][:])
    else:
        shape = TensorShape(attrs['relay_shape'][:])

    logger.debug("-- newshape: {}".format(shape))

    return {'shape': shape}


#############
# AnyOp #
#############

@xop_register('AnyOp')
def any_op(attrs: Dict, in_xlayers: List[XLayer]):
    """
    Create an AnyOp. This operation can have any number of inputs and
    attributes and returns one output. Only the 'any_shape' attribute
    is required to generate an operation shape

    Attributes:
    -----------
    op_name: str
        The name of the operation
    in_xlayers: List[XLayer]
        A list of the input_layers
    any_shape: List[int] / List[List[int]]
        The shape of the operation
    """

    shape = attrs['any_shape']
    if len(shape) > 0 and isinstance(shape[0], list):
        shape = TupleShape(attrs['any_shape'][:])
    else:
        shape = TensorShape(attrs['any_shape'][:])

    logger.debug("--anyshape: {}".format(shape))

    return {'shape': shape}
