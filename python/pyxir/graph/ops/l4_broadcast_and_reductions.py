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

L4: Broadcast and Reduction Operators
"""


import math
import logging
import warnings
import numpy as np

from pyxir.shapes import TensorShape

from ..layer.xlayer import defaultXLayer, XLayer
from ..layer.xlayer_factory import xop_register_factory
from ..xop_registry import xop_register_op_layout_transform,\
    xop_register_op_transpose_transform

logger = logging.getLogger("pyxir")


########
# Mean #
########

@xop_register_factory('Mean')
def mean(op_name, input_layer, axes, keepdims, exclude, **kwargs):
    # type: (str, XLayer, List[int], boolean, List[int]) -> XLayer
    """
    Compute the mean of the input layer over some axes

    Arguments
    ---------
    op_name: str
        The name of this elementwise addition operation
    axes: List[int]
        The axes over which to compute the mean
    ... TODO
    input_layer: XLayer
        The input layer
    """

    attrs = kwargs

    logger.debug("Attrs: {}".format(attrs))

    bottoms = [input_layer.name]

    in_shape = input_layer.shapes[:]

    if exclude:
        axes = [i for i in range(len(in_shape)) if i not in axes]

    if keepdims:
        newshape = [dim if i not in axes else 1
                    for i, dim in enumerate(in_shape)]
    else:
        newshape = [dim for i, dim in enumerate(in_shape)
                    if i not in axes]

    newshape = TensorShape(newshape)
    logger.debug("Mean axes: {}, in shape: {}, out shape: {}"
                 .format(axes, in_shape, newshape))

    attrs.update({
        'axes': axes,
        'keepdims': keepdims,
        # 'exclude': exclude
        #  TODO: dtype??
    })

    X = XLayer()
    X = X._replace(
        name=op_name,
        type=['Mean'],
        shapes=newshape,
        sizes=newshape.get_size(),
        layer=[op_name],
        tops=[],
        bottoms=bottoms,
        attrs=attrs,
        targets=[]
    )

    return X


@xop_register_op_transpose_transform('Mean')
def mean_transpose_transform(X, axes):
    # type: (XLayer, List[int]) -> None
    """ Transform Mean layer with transpose according to provided axes """

    new_shape = [X.shapes[i] for i in axes]
    X.shapes = new_shape
    X.attrs['axes'] = [axes.index(axis) for axis in X.attrs['axes']]
