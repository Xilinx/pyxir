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
Data structure for registering and tracking XLayer to Tensorflow layer
conversion functions


"""

import copy
import logging

from pyxir.shapes import TensorShape, TupleShape

from . import rt_layer_tf
from .. import base

logger = logging.getLogger("pyxir")


X_2_TF = {}


def rt_register_xlayer_2_tf(xop_name):
    # type: (str) -> function
    """ Return decorator for registering XLayer to Tensorflow conversion
        for Tensorflow runtime"""

    def __rt_register_xlayer_2_tf(Cls):

        def get_layer(X, input_shapes, params, **kwargs):
            # (XLayer, dict, dict) -> List[rt_layer.RtLayer]
            """ Generic function for constructing a TF layer from an XLayer"""
            shapes = X.shapes[:]

            return [Cls(
                name=X.name,
                xtype=X.type[0],
                shape=shapes,
                dtype=X.attrs['dtype'] if 'dtype' in X.attrs else 'float32',
                inputs=X.bottoms[:],
                input_shapes=[input_shapes[bottom]
                              for bottom in X.bottoms],
                data=X.data,
                subgraph=X.subgraph,
                attrs=copy.deepcopy(X.attrs)
            )]

        if xop_name in X_2_TF:
            raise ValueError("Cant't register XLayer to Tensorflow factory "
                             " function: {} as it has already been registered"
                             .format(xop_name))

        X_2_TF[xop_name] = get_layer

        return Cls

    return __rt_register_xlayer_2_tf


def rt_register_xlayer_2_tf_factory_func(xop_name):
    # type: (str) -> function
    """ Return decorator for registering XLayer to Tensorflow conversion
        for Tensorflow runtime"""

    def __rt_register_xlayer_2_tf_factory_func(factory_func):
        if xop_name in X_2_TF:
            raise ValueError("Cant't register XLayer to Tensorflow factory "
                             " function: {} as it has already been registered"
                             .format(xop_name))

        X_2_TF[xop_name] = factory_func()

    return __rt_register_xlayer_2_tf_factory_func
