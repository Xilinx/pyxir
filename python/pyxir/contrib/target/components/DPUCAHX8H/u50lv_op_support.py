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
Module for declaring and specifying supported operations for the DPUCAHX8H-u50lv target.
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
    is_upscale_supported,
)

logger = logging.getLogger("pyxir")


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "BatchNorm")
def batchnorm_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided BatchNorm operator
    on the DPUCAHX8H-u50lv target"""
    return is_batch_norm_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "BiasAdd")
def biasadd_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided BiasAdd operator
    on the DPUCAHX8H-u50lv target"""
    return is_bias_add_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Cast")
def cast_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Cast operator
    on the DPUCAHX8H-u50lv target"""
    dtype = X.attrs["dtype"]
    return dtype == "float32"


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Concat")
def concat_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Concat operator
    on the DPUCAHX8H-u50lv target"""
    return is_concat_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Convolution")
def conv2d_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Conv2D operator
    on the DPUCAHX8H-u50lv target"""
    return is_conv2d_supported(
        X,
        bXs,
        tXs,
        channel_parallel=16,
        bank_depth=2048,
        max_stride=4,
        min_stride=1,
        depthwise_supported=False,
    )


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Conv2DTranspose")
def conv2d_transpose_op_support(
    X: XLayer, bXs: List[XLayer], tXs: List[XLayer]
) -> bool:
    """Check whether we can execute the provided Conv2DTranspose operator
    on the DPUCAHX8H-u50lv target"""
    return is_conv2d_transpose_supported(
        X, bXs, tXs, channel_parallel=16, bank_depth=2048, max_stride=16, min_stride=1
    )


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "DPU")
def DPUCZDX8G_kv260_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided DPU operator
    on the DPUCAHX8H-u50lv target"""
    return True


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Eltwise")
def eltwise_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Eltwise operator
    on the DPUCAHX8H-u50lv target"""
    return True


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Maximum")
def maximum_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Maximum operator
        on the DPUCAHX8H-u50lv target
    Return true if part of leaky relu pattern
    """
    # check whether part of leaky relu
    return "patterns" in X.attrs and "LeakyReLU" in X.attrs["patterns"]


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Pad")
def pad_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Padding operator
    on the DPUCAHX8H-u50lv target"""
    return is_padding_supported(X, bXs, tXs)


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Pooling")
def pooling_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Pooling operator
    on the DPUCAHX8H-u50lv target"""
    return is_pooling_supported(
        X=X,
        bXs=bXs,
        tXs=tXs,
        channel_parallel=16,
        bank_depth=2048,
        max_pool_min_kernel=1,
        max_pool_max_kernel=8,
        max_pool_min_stride=1,
        max_pool_max_stride=8,
        avg_pool_min_kernel=1,
        avg_pool_max_kernel=8,
        avg_pool_min_stride=1,
        avg_pool_max_stride=8,
    )


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Mean")
def mean_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Mean operator
    on the DPUCAHX8H-u50lv target"""
    axes = X.attrs["axes"]
    keepdims = X.attrs["keepdims"]
    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "LeakyReLU")
def leaky_relu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided LeakyRelu operator
    on the DPUCAHX8H-u50lv target"""
    assert len(bXs) == 1
    bX = bXs[0]
    alpha = X.attrs["alpha"]
    # LeakyRelu not supported after depthwise conv2d
    return math.isclose(alpha, 0.1, rel_tol=1e-5) and bX.type[0] in [
        "Convolution",
        "Conv2DTranspose",
        "BatchNorm",
        "BiasAdd",
        "Scale",
    ]


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "pReLU")
def prelu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided pRelu operator
    on the DPUCAHX8H-u50lv target"""
    # Only LeakyReLU: alpha == 0.1 supported
    assert len(bXs) == 1
    bX = bXs[0]
    alpha = X.attrs["alpha"]
    # LeakyRelu not supported after depthwise conv2d
    return math.isclose(alpha, 0.1, rel_tol=1e-5) and bX.type[0] in [
        "Convolution",
        "Conv2DTranspose",
        "BatchNorm",
        "BiasAdd",
        "Scale",
    ]


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "ReLU")
def relu_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided ReLU operator
    on the DPUCAHX8H-u50lv target"""
    assert len(bXs) == 1
    bX = bXs[0]
    return bX.type[0] in set(
        ["Convolution", "Conv2DTranspose", "Eltwise", "BatchNorm", "BiasAdd", "Scale"]
    )


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "ReLU6")
def relu6_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided ReLU6 operator
    on the DPUCAHX8H-u50lv target"""
    assert len(bXs) == 1
    bX = bXs[0]
    return bX.type[0] in set(
        ["Convolution", "Conv2DTranspose", "BatchNorm", "Scale"]
    )


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Scale")
def scale_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Scale operator
    on the DPUCAHX8H-u50lv target"""
    return is_scale_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048)


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Upsampling2D")
def upsampling_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Upsampling2D operator
    on the DPUCAHX8H-u50lv target"""
    return is_upscale_supported(X, bXs, tXs, channel_parallel=16, bank_depth=2048, bank_num=8)


@pyxir.register_op_support_check("DPUCAHX8H-u50lv", "Dropout")
def dropout_op_support(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    """Check whether we can execute the provided Dropout operator
    on the DPUCAHX8H-u50lv target"""
    return True
