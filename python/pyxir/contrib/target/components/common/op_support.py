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

"""Module for common operation support check functionality"""

import math
from typing import List

from pyxir.graph import XLayer


def is_batch_norm_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    axis = X.attrs["axis"]
    channels = X.shapes[axis]
    return channels >= 1 and channels <= 256 * channel_parallel


def is_bias_add_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    axis = X.attrs["axis"]
    channels = X.shapes[axis]
    return channels >= 1 and channels <= 256 * channel_parallel


def is_concat_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    axis = X.attrs["axis"]
    channels = X.shapes[axis]
    return channels >= 1 and channels <= 256 * channel_parallel


def is_conv2d_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
    max_stride: int,
    min_stride: int,
    max_kernel_sz: int = 16,
    min_kernel_sz: int = 1,
    depthwise_supported: bool = True,
) -> bool:
    data_layout = X.attrs["data_layout"]
    kernel_h, kernel_w = X.attrs["kernel_size"]
    stride_h, stride_w = X.attrs["strides"]
    dilation_h, dilation_w = X.attrs["dilation"]
    padding_h, padding_w = (
        X.attrs["padding"][data_layout.index("H")],
        X.attrs["padding"][data_layout.index("W")],
    )
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    ch_in, ch_out = X.attrs["channels"]
    groups = X.attrs["groups"]

    return (
        kernel_h >= min_kernel_sz
        and kernel_h <= max_kernel_sz
        and kernel_w >= min_kernel_sz
        and kernel_w <= max_kernel_sz
        and stride_h >= min_stride
        and stride_h <= max_stride
        and stride_w >= min_stride
        and stride_w <= max_stride
        and padding_h_top >= 0
        and padding_h_top <= (kernel_h - 1) * dilation_h + 1
        and padding_h_bot >= 0
        and padding_h_bot <= (kernel_h - 1) * dilation_h + 1
        and padding_w_left >= 0
        and padding_w_left <= (kernel_w - 1) * dilation_w + 1
        and padding_w_right >= 0
        and padding_w_right <= (kernel_w - 1) * dilation_w + 1
        and ch_in >= 1
        and ch_in <= 256 * channel_parallel
        and ch_out >= 1
        and ch_out <= 256 * channel_parallel
        and dilation_h * ch_in <= 256 * channel_parallel
        and (dilation_h == 1 or stride_h == 1)
        and dilation_w * ch_in <= 256 * channel_parallel
        and (dilation_w == 1 or stride_w == 1)
        and kernel_w * kernel_h * math.ceil(ch_in / channel_parallel) <= bank_depth
        and depthwise_supported
        or groups == 1
    )


def is_conv2d_transpose_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
    max_stride: int,
    min_stride: int,
    max_kernel_sz: int = 16,
    min_kernel_sz: int = 1,
) -> bool:
    data_layout = X.attrs["data_layout"]

    kernel_h, kernel_w = X.attrs["kernel_size"]
    stride_h, stride_w = X.attrs["strides"]
    dilation_h, dilation_w = X.attrs["dilation"]
    padding_h, padding_w = (
        X.attrs["padding"][data_layout.index("H")],
        X.attrs["padding"][data_layout.index("W")],
    )
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    ch_in, ch_out = X.attrs["channels"]
    groups = X.attrs["groups"]

    return (
        kernel_h >= min_kernel_sz
        and kernel_h <= max_kernel_sz
        and kernel_w >= min_kernel_sz
        and kernel_w <= max_kernel_sz
        and stride_h >= min_stride
        and stride_h <= max_stride
        and stride_w >= min_stride
        and stride_w <= max_stride
        and padding_h_top >= 0
        and padding_h_top <= kernel_h - 1
        and padding_h_bot >= 0
        and padding_h_bot <= kernel_h - 1
        and padding_w_left >= 0
        and padding_w_left <= kernel_w - 1
        and padding_w_right >= 0
        and padding_w_right <= kernel_w - 1
        and ch_in >= 1
        and ch_in <= 256 * channel_parallel
        and ch_out >= 1
        and ch_out <= 256 * channel_parallel
        and dilation_h * ch_in <= 256 * channel_parallel
        and (dilation_h == 1 or stride_h == 1)
        and dilation_w * ch_in <= 256 * channel_parallel
        and (dilation_w == 1 or stride_w == 1)
    )


def is_padding_supported(X: XLayer, bXs: List[XLayer], tXs: List[XLayer]) -> bool:

    if len(tXs) == 1 and tXs[0].type[0] in ["Pooling", "Convolution"]:
        t_data_layout = tXs[0].attrs["data_layout"]
        t_type = tXs[0].type[0]

        padding_h, padding_w = (
            X.attrs["padding"][t_data_layout.index("H")],
            X.attrs["padding"][t_data_layout.index("W")],
        )
        padding_h_top, padding_h_bot = padding_h
        padding_w_left, padding_w_right = padding_w

        if t_type == "Pooling":
            t_kernel_h, t_kernel_w = tXs[0].attrs["kernel_size"]
            return (
                padding_h_top >= 0
                and padding_h_top <= t_kernel_h - 1
                and padding_h_bot >= 0
                and padding_h_bot <= t_kernel_h - 1
                and padding_w_left >= 0
                and padding_w_left <= t_kernel_w - 1
                and padding_w_right >= 0
                and padding_w_right <= t_kernel_w - 1
            )
        elif t_type == "Convolution":
            t_kernel_h, t_kernel_w = tXs[0].attrs["kernel_size"]
            t_dilation_h, t_dilation_w = tXs[0].attrs["dilation"]

            return (
                padding_h_top >= 0
                and padding_h_top <= (t_kernel_h - 1) * t_dilation_h + 1
                and padding_h_bot >= 0
                and padding_h_bot <= (t_kernel_h - 1) * t_dilation_h + 1
                and padding_w_left >= 0
                and padding_w_left <= (t_kernel_w - 1) * t_dilation_w + 1
                and padding_w_right >= 0
                and padding_w_right <= (t_kernel_w - 1) * t_dilation_w + 1
            )

        return False

    return False


def is_pooling_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
    max_pool_min_kernel: int,
    max_pool_max_kernel: int,
    max_pool_min_stride: int,
    max_pool_max_stride: int,
    avg_pool_min_kernel: int,
    avg_pool_max_kernel: int,
    avg_pool_min_stride: int,
    avg_pool_max_stride: int,
    max_pool_kernel_valid: List[int] = None,
    avg_pool_kernel_valid: List[int] = None,
) -> bool:
    pool_type = X.attrs["pool_type"]
    data_layout = X.attrs["data_layout"]
    kernel_h, kernel_w = X.attrs["kernel_size"]
    stride_h, stride_w = X.attrs["strides"]
    padding_h, padding_w = (
        X.attrs["padding"][data_layout.index("H")],
        X.attrs["padding"][data_layout.index("W")],
    )
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    channels = X.shapes[data_layout.index("C")]

    if pool_type == "Max":
        return (
            kernel_h >= max_pool_min_kernel
            and kernel_h <= max_pool_max_kernel
            and kernel_w >= max_pool_min_kernel
            and kernel_w <= max_pool_max_kernel
            and (max_pool_kernel_valid is None or kernel_h in max_pool_kernel_valid)
            and (max_pool_kernel_valid is None or kernel_w in max_pool_kernel_valid)
            and stride_h >= max_pool_min_stride
            and stride_h <= max_pool_max_stride
            and stride_w >= max_pool_min_stride
            and stride_w <= max_pool_max_stride
            and padding_h_top >= 0
            and padding_h_top <= (kernel_h - 1)
            and padding_h_bot >= 0
            and padding_h_bot <= (kernel_h - 1)
            and padding_w_left >= 0
            and padding_w_left <= (kernel_w - 1)
            and padding_w_right >= 0
            and padding_w_right <= (kernel_w - 1)
            and channels >= 1
            and channels <= channel_parallel * 256
        )
    elif pool_type == "Avg":
        return (
            kernel_h == kernel_w
            and kernel_h >= avg_pool_min_kernel
            and kernel_h <= avg_pool_max_kernel
            and (avg_pool_kernel_valid is None or kernel_h in avg_pool_kernel_valid)
            and (avg_pool_kernel_valid is None or kernel_w in avg_pool_kernel_valid)
            and stride_h >= avg_pool_min_stride
            and stride_h <= avg_pool_max_stride
            and stride_w >= avg_pool_min_stride
            and stride_w <= avg_pool_max_stride
            and padding_h_top >= 0
            and padding_h_top <= (kernel_h - 1)
            and padding_h_bot >= 0
            and padding_h_bot <= (kernel_h - 1)
            and padding_w_left >= 0
            and padding_w_left <= (kernel_w - 1)
            and padding_w_right >= 0
            and padding_w_right <= (kernel_w - 1)
            and channels >= 1
            and channels <= channel_parallel * 256
        )
    return False


def is_scale_supported(
    X: XLayer,
    bXs: List[XLayer],
    tXs: List[XLayer],
    channel_parallel: int,
    bank_depth: int,
) -> bool:
    axis = X.attrs["axis"]
    channels = X.shapes[axis]
    return channels >= 1 and channels <= 256 * channel_parallel
