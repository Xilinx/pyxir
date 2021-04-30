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
Module for declaring and specifying supported operations for
DPUCZDX8G zcu102 target.
"""

import math
import logging
import pyxir

from typing import List

from pyxir.graph import XLayer

from .common_op_support import (
    is_batch_norm_supported,
    is_bias_add_supported,
    is_concat_supported,
    is_conv2d_supported,
    is_conv2d_transpose_supported,
    is_padding_supported,
    is_pooling_supported,
    is_relu_supported,
    is_relu6_supported,
    is_leaky_relu_supported,
    is_scale_supported,
)

logger = logging.getLogger("pyxir")


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "BatchNorm")
def batchnorm_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided BatchNorm operator
        on the zcu102 target """
    return is_batch_norm_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "BiasAdd")
def biasadd_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided BiasAdd operator
        on the zcu102 target """
    return is_bias_add_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Cast")
def cast_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Cast operator
        on the DPUCZDX8G-zcu102 target """
    dtype = X.attrs["dtype"]
    return dtype == "float32"


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Concat")
def concat_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Concat operator
        on the zcu102 target """
    return is_concat_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Convolution")
def conv2d_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Conv2D operator
        on the zcu102 target """
    channel_parallel = 16
    bank_depth = 2048
    return is_conv2d_supported(X, bXs, tXs, channel_parallel, bank_depth)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Conv2DTranspose")
def conv2d_transpose_op_support(
    X: XLayer, bXs: List[XLayer], tXs: List[XLayer]
) -> bool:
    """ Check whether we can execute the provided Conv2DTranspose operator
        on the zcu102 target """
    channel_parallel = 16
    bank_depth = 2048
    return is_conv2d_transpose_supported(X, bXs, tXs, channel_parallel, bank_depth)


# @pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Dense')
# def dense_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
#
#     """ Check whether we can execute the provided Dense operator
#         on the zcu102 target """

#     # TODO out_ch

#     return True


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "DPU")
def DPUCZDX8G_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided DPU operator
        on the zcu102 target """

    return True


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Eltwise")
def eltwise_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Eltwise operator
        on the zcu102 target """

    # TODO in_ch

    return True


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Maximum")
def maximum_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Maximum operator
        on the zcu102 target

    Return true if part of leaky relu pattern    
    """
    # check whether part of leaky relu
    return "patterns" in X.attrs and "LeakyReLU" in X.attrs["patterns"]


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Pad")
def pad_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Pooling operator
        on the zcu102 target """
    return is_padding_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Pooling")
def pooling_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Pooling operator
        on the zcu102 target """
    return is_pooling_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Mean")
def mean_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Mean operator
        on the zcu102 target """
    axes = X.attrs["axes"]
    keepdims = X.attrs["keepdims"]
    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "LeakyReLU")
def leaky_relu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided LeakyRelu operator
        on the zcu102 target """
    return is_leaky_relu_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "pReLU")
def prelu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided pRelu operator
        on the zcu102 target """
    # Only LeakyReLU: alpha == 0.1 supported
    return is_leaky_relu_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "ReLU")
def relu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided ReLU operator
        on the zcu102 target """
    return is_relu_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "ReLU6")
def relu6_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided ReLU operator
        on the zcu102 target """
    return is_relu6_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Scale")
def scale_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Scale operator
        on the zcu102 target """
    return is_scale_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCZDX8G-zcu102", "Upsampling2D")
def upsampling_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Upsampling2D operator
       on the zcu102 target"""

    method = X.attrs["method"]
    # TODO
    return method == "nearest_neighbor"
