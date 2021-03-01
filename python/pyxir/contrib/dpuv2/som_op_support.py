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
DPU V2 som target.

NOTE: https://www.xilinx.com/support/documentation/ip_documentation/dpu/v3_0/pg338-dpu.pdf # noqa
"""

import math
import logging

import pyxir

logger = logging.getLogger('pyxir')


@pyxir.register_op_support_check('dpuv2-som', 'BatchNorm')
def batchnorm_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BatchNorm operator
        on the som target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 3136


@pyxir.register_op_support_check('dpuv2-som', 'BiasAdd')
def biasadd_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BiasAdd operator
        on the som target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 3136


@pyxir.register_op_support_check('dpuv2-som', 'Cast')
def cast_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Cast operator
        on the dpuv2-som target """

    dtype = X.attrs['dtype']

    return dtype == 'float32'


@pyxir.register_op_support_check('dpuv2-som', 'Concat')
def concat_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Concat operator
        on the som target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 3136


@pyxir.register_op_support_check('dpuv2-som', 'Convolution')
def conv2d_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2D operator
        on the som target """

    data_layout = X.attrs['data_layout']

    kernel_h, kernel_w = X.attrs['kernel_size']
    stride_h, stride_w = X.attrs['strides']
    dilation_h, dilation_w = X.attrs['dilation']
    padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
        X.attrs['padding'][data_layout.index('W')]
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    ch_in, ch_out = X.attrs['channels']
    groups = X.attrs['groups']

    return kernel_h >= 1 and kernel_h <= 16 and\
        kernel_w >= 1 and kernel_w <= 16 and\
        stride_h >= 1 and stride_h <= 4 and\
        stride_w >= 1 and stride_w <= 4 and\
        padding_h_top >= 0 and padding_h_top <= kernel_h - 1 and\
        padding_h_bot >= 0 and padding_h_bot <= kernel_h - 1 and\
        padding_w_left >= 0 and padding_w_left <= kernel_w - 1 and\
        padding_w_right >= 0 and padding_w_right <= kernel_w - 1 and\
        ch_in*groups >= 1 and ch_in*groups <= 3136 and\
        ch_out >= 1 and ch_out <= 3136 and\
        dilation_h * ch_in <= 3136 and\
        (dilation_h == 1 or stride_h == 1) and\
        dilation_w * ch_in <= 3136 and\
        (dilation_w == 1 or stride_w == 1)


@pyxir.register_op_support_check('dpuv2-som', 'Conv2DTranspose')
def conv2d_transpose_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2DTranspose operator
        on the som target """

    data_layout = X.attrs['data_layout']

    kernel_h, kernel_w = X.attrs['kernel_size']
    stride_h, stride_w = X.attrs['strides']
    dilation_h, dilation_w = X.attrs['dilation']
    padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
        X.attrs['padding'][data_layout.index('W')]
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    ch_in, ch_out = X.attrs['channels']
    groups = X.attrs['groups']

    return kernel_h >= 1 and kernel_h <= 16 and\
        kernel_w >= 1 and kernel_w <= 16 and\
        stride_w * ch_out >= 1 and stride_w * ch_out <= 3136 and\
        stride_h >= 1 and\
        padding_h_top >= 0 and padding_h_top <= kernel_h - 1 and\
        padding_h_bot >= 0 and padding_h_bot <= kernel_h - 1 and\
        padding_w_left >= 0 and padding_w_left <= kernel_w - 1 and\
        padding_w_right >= 0 and padding_w_right <= kernel_w - 1 and\
        ch_in*groups >= 1 and ch_in*groups <= 3136 and\
        ch_out >= 1 and ch_out <= 3136 and\
        dilation_h * ch_in <= 3136 and\
        (dilation_h == 1 or stride_h == 1) and\
        dilation_w * ch_in <= 3136 and\
        (dilation_w == 1 or stride_w == 1)


# @pyxir.register_op_support_check('dpuv2-som', 'Dense')
# def dense_op_support(X, bXs, tXs):
#     # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
#     """ Check whether we can execute the provided Dense operator
#         on the som target """

#     # TODO out_ch

#     return True


@pyxir.register_op_support_check('dpuv2-som', 'DPU')
def dpu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided DPU operator
        on the som target """

    # TODO out_ch

    return True


@pyxir.register_op_support_check('dpuv2-som', 'Eltwise')
def eltwise_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Eltwise operator
        on the som target """

    # TODO in_ch

    return True


@pyxir.register_op_support_check('dpuv2-som', 'Pad')
def pad_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the som target """

    if len(tXs) == 1 and tXs[0].type[0] in ['Pooling', 'Convolution']:
        t_data_layout = tXs[0].attrs['data_layout']
        t_type = tXs[0].type[0]

        padding_h, padding_w = X.attrs['padding'][t_data_layout.index('H')],\
            X.attrs['padding'][t_data_layout.index('W')]
        padding_h_top, padding_h_bot = padding_h
        padding_w_left, padding_w_right = padding_w

        if t_type == 'Pooling':
            return padding_h_top >= 0 and padding_h_top <= 4 and\
                padding_h_bot >= 0 and padding_h_bot <= 4 and\
                padding_w_left >= 0 and padding_w_left <= 4 and\
                padding_w_right >= 0 and padding_w_right <= 4
        elif t_type == 'Convolution':
            t_kernel_h, t_kernel_w = tXs[0].attrs['kernel_size']

            return padding_h_top >= 0 and padding_h_top <= t_kernel_h - 1 and\
                padding_h_bot >= 0 and padding_h_bot <= t_kernel_h - 1 and\
                padding_w_left >= 0 and padding_w_left <= t_kernel_w - 1 and\
                padding_w_right >= 0 and padding_w_right <= t_kernel_w - 1

        return False

    return False


@pyxir.register_op_support_check('dpuv2-som', 'Pooling')
def pooling_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the som target """

    data_layout = X.attrs['data_layout']

    kernel_h, kernel_w = X.attrs['kernel_size']
    stride_h, stride_w = X.attrs['strides']
    padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
        X.attrs['padding'][data_layout.index('W')]
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w

    channels = X.shapes[data_layout.index('C')]

    return kernel_h >= 1 and kernel_h <= 8 and\
        kernel_w >= 1 and kernel_w <= 8 and\
        stride_h >= 1 and stride_h <= 4 and\
        stride_w >= 1 and stride_w <= 4 and\
        padding_h_top >= 0 and padding_h_top <= 4 and\
        padding_h_bot >= 0 and padding_h_bot <= 4 and\
        padding_w_left >= 0 and padding_w_left <= 4 and\
        padding_w_right >= 0 and padding_w_right <= 4 and\
        channels >= 1 and channels <= 3136


@pyxir.register_op_support_check('dpuv2-som', 'Mean')
def mean_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Mean operator
        on the som target """

    axes = X.attrs['axes']
    keepdims = X.attrs['keepdims']

    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check('dpuv2-som', 'LeakyReLU')
def leaky_relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided LeakyRelu operator
        on the som target """

    # TODO: position?

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('dpuv2-som', 'pReLU')
def prelu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided pRelu operator
        on the som target """

    # TODO: position?

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('dpuv2-som', 'ReLU')
def relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the som target """

    # TODO always?

    return True


@pyxir.register_op_support_check('dpuv2-som', 'ReLU6')
def relu6_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the som target """

    # TODO always?

    return True


@pyxir.register_op_support_check('dpuv2-som', 'Scale')
def scale_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Scale operator
        on the som target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels > 1 and channels <= 3136


@pyxir.register_op_support_check('dpuv2-som', 'Upsampling2D')
def scale_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Upsampling2D operator
        on the som target """

    method = X.attrs['method']
    # TODO
    return method == 'nearest_neighbor'
