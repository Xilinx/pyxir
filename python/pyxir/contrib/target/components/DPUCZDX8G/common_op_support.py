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

"""Module for common DPUCZDX8G operation support functionality"""

import math

from typing import List

from pyxir.graph import XLayer

from pyxir.contrib.target.components.common.op_support import (
    is_batch_norm_supported,
    is_bias_add_supported,
    is_concat_supported,
    is_conv2d_supported as is_conv2d_supported_common,
    is_conv2d_transpose_supported as is_conv2d_transpose_supported_common,
    is_padding_supported,
    is_pooling_supported as is_pooling_supported_common,
    is_scale_supported,
)


def is_conv2d_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    return is_conv2d_supported_common(
        X,
        bXs,
        tXs,
        channel_parallel=channel_parallel,
        bank_depth=bank_depth,
        max_stride=8,
        min_stride=1,
        max_kernel_sz=16,
        min_kernel_sz=1,
    )


def is_conv2d_transpose_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    return is_conv2d_transpose_supported_common(
        X,
        bXs,
        tXs,
        channel_parallel=channel_parallel,
        bank_depth=bank_depth,
        max_stride=16,
        min_stride=1,
        max_kernel=16,
        min_kernel=1,
    )


def is_pooling_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    return is_pooling_supported_common(
        X,
        bXs,
        tXs,
        channel_parallel=channel_parallel,
        bank_depth=bank_depth,
        max_pool_min_kernel=2,
        max_pool_max_kernel=8,
        max_pool_min_stride=1,
        max_pool_max_stride=8,
        avg_pool_min_kernel=2,
        avg_pool_max_kernel=8,
        avg_pool_min_stride=1,
        avg_pool_max_stride=8,
    )


def is_relu_supported(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    assert len(bXs) == 1
    bX = bXs[0]
    return bX.type[0] in set(
        ["Convolution", "Conv2DTranspose", "Pooling", "Eltwise", "BatchNorm", "Scale"]
    )


def is_relu6_supported(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    assert len(bXs) == 1
    bX = bXs[0]
    return bX.type[0] in set(["Convolution", "Conv2DTranspose", "BatchNorm", "Scale"])


def is_leaky_relu_supported(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:
    assert len(bXs) == 1
    bX = bXs[0]
    alpha = X.attrs["alpha"]
    # LeakyRelu not supported after depthwise conv2d
    return (
        math.isclose(alpha, 0.1, rel_tol=1e-5)
        and bX.type[0] in ["Convolution", "Conv2DTranspose", "BatchNorm", "Pooling"]
        # and bX.type[0] != "Convolution"
        # or bX.attrs["groups"] == 1
    )