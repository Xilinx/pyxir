# Copyright 2021 Xilinx Inc.
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
Module for declaring and specifying supported operations for the DPUCAHX8L target.
See https://www.xilinx.com/html_docs/vitis_ai/1_3/compiling_model.html#ztl1570696058091
"""

import math
import logging
import pyxir

from typing import List

from pyxir.graph import XLayer
from pyxir.contrib.target.components.common.op_support import (
    is_batch_norm_supported,
    is_bias_add_supported,
    is_concat_supported,
    is_conv2d_supported,
    is_conv2d_transpose_supported,
    is_padding_supported,
    is_pooling_supported,
    is_scale_supported,
)

logger = logging.getLogger("pyxir")


@pyxir.register_op_support_check("DPUCAHX8L", "BatchNorm")
def batchnorm_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided BatchNorm operator
        on the DPUCAHX8L target """
    return is_batch_norm_supported(X, bXs, tXs, channel_parallel=32, bank_depth=4096)


@pyxir.register_op_support_check("DPUCAHX8L", "BiasAdd")
def biasadd_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided BiasAdd operator
        on the DPUCAHX8L target """
    return is_bias_add_supported(X, bXs, tXs, channel_parallel=32, bank_depth=4096)


@pyxir.register_op_support_check("DPUCAHX8L", "Cast")
def cast_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Cast operator
        on the DPUCAHX8L target """
    dtype = X.attrs["dtype"]
    return dtype == "float32"


@pyxir.register_op_support_check("DPUCAHX8L", "Concat")
def concat_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Concat operator
        on the DPUCAHX8L target """
    return is_concat_supported(X, bXs, tXs, channel_parallel=32, bank_depth=4096)


@pyxir.register_op_support_check("DPUCAHX8L", "Convolution")
def conv2d_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Conv2D operator
        on the DPUCAHX8L target """
    return is_conv2d_supported(
        X,
        bXs,
        tXs,
        channel_parallel=32,
        bank_depth=4096,
        max_stride=4,
        min_stride=1,
        depthwise_supported=True,
    )


@pyxir.register_op_support_check("DPUCAHX8L", "Conv2DTranspose")
def conv2d_transpose_op_support(
    X: XLayer, bXs: List[XLayer], tXs: List[XLayer]
) -> bool:
    """Check whether we can execute the provided Conv2DTranspose operator
       on the DPUCAHX8L target"""
    return is_conv2d_transpose_supported(
        X, bXs, tXs, channel_parallel=32, bank_depth=4096, max_stride=16, min_stride=1
    )


@pyxir.register_op_support_check("DPUCAHX8L", "DPU")
def DPUCZDX8G_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided DPU operator
        on the DPUCAHX8L target """
    return True


@pyxir.register_op_support_check("DPUCAHX8L", "Eltwise")
def eltwise_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Eltwise operator
       on the DPUCAHX8L target"""
    return True


@pyxir.register_op_support_check("DPUCAHX8L", "Maximum")
def maximum_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Maximum operator
        on the DPUCAHX8L target
    Return true if part of leaky relu pattern    
    """
    # check whether part of leaky relu
    return "patterns" in X.attrs and "LeakyReLU" in X.attrs["patterns"]


@pyxir.register_op_support_check("DPUCAHX8L", "Pad")
def pad_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Padding operator
        on the DPUCAHX8L target """
    return is_padding_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCAHX8L", "Pooling")
def pooling_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Pooling operator
        on the DPUCAHX8L target """
    return is_pooling_supported(
        X=X,
        bXs=bXs,
        tXs=tXs,
        channel_parallel=32,
        bank_depth=4096,
        max_pool_min_kernel=2,
        max_pool_max_kernel=8,
        max_pool_min_stride=1,
        max_pool_max_stride=8,
        avg_pool_min_kernel=1,
        avg_pool_max_kernel=8,
        avg_pool_min_stride=1,
        avg_pool_max_stride=8,
        max_pool_kernel_valid=[2, 3, 5, 7, 8],
        avg_pool_kernel_valid=[2, 3, 5, 7, 8],
    )


@pyxir.register_op_support_check("DPUCAHX8L", "Mean")
def mean_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Mean operator
        on the DPUCAHX8L target """
    axes = X.attrs["axes"]
    keepdims = X.attrs["keepdims"]
    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check("DPUCAHX8L", "LeakyReLU")
def leaky_relu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided LeakyRelu operator
        on the DPUCAHX8L target """
    return False


@pyxir.register_op_support_check("DPUCAHX8L", "pReLU")
def prelu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided pRelu operator
        on the DPUCAHX8L target """
    return False


@pyxir.register_op_support_check("DPUCAHX8L", "ReLU")
def relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the zcu104 target """
    assert len(bXs) == 1
    bX = bXs[0]
    return bX.type[0] in set(
        ["Convolution", "Conv2DTranspose", "Eltwise", "BatchNorm", "Scale"]
    )


@pyxir.register_op_support_check("DPUCAHX8L", "ReLU6")
def relu6_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided ReLU6 operator
        on the DPUCAHX8L target """
    return True


@pyxir.register_op_support_check("DPUCAHX8L", "Scale")
def scale_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Scale operator
        on the DPUCAHX8L target """
    return is_scale_supported(X, bXs, tXs, channel_parallel=32, bank_depth=4096)


@pyxir.register_op_support_check("DPUCAHX8L", "Upsampling2D")
def upsampling_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """ Check whether we can execute the provided Upsampling2D operator
        on the DPUCAHX8L target """
    # TODO
    method = X.attrs["method"]
    return method == "nearest_neighbor"
