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
Module for declaring and specifying supported operations for DPU v1 target.

NOTE: https://gitenterprise.xilinx.com/jornt/MLsuite/blob/master/docs/ml-suite-overview.md # noqa
"""

import math
import pyxir
import logging

logger = logging.getLogger('pyxir')


@pyxir.register_op_support_check('DPUCADX8G', 'BatchNorm')
def batchnorm_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BatchNorm operator
        on the DPUCADX8G target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADX8G', 'BiasAdd')
def biasadd_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BiasAdd operator
        on the DPUCADX8G target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADX8G', 'Cast')
def cast_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Cast operator
        on the DPUCADX8G target """

    dtype = X.attrs['dtype']

    return dtype == 'float32'

@pyxir.register_op_support_check('DPUCADX8G', 'Concat')
def concat_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Concat operator
        on the DPUCADX8G target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADX8G', 'Convolution')
def conv2d_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2D operator
        on the DPUCADX8G target """

    data_layout = X.attrs['data_layout']

    kernel_h, kernel_w = X.attrs['kernel_size']
    stride_h, stride_w = X.attrs['strides']
    dilation_h, dilation_w = X.attrs['dilation']
    padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
        X.attrs['padding'][data_layout.index('H')]
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    ch_in, ch_out = X.attrs['channels']
    groups = X.attrs['groups']

    # TODO padding?
    # padding_h_top >= 0 and padding_h_top <= kernel_h - 1 and\
    # padding_h_bot >= 0 and padding_h_bot <= kernel_h - 1 and\
    # padding_w_left >= 0 and padding_w_left <= kernel_w - 1 and\
    # padding_w_right >= 0 and padding_w_right <= kernel_w - 1 and\
    return groups == 1 and\
        kernel_h >= 1 and kernel_h <= 15 and\
        kernel_w >= 1 and kernel_w <= 15 and\
        stride_h in [1, 2, 4, 8] and\
        stride_w in [1, 2, 4, 8] and\
        ch_in >= 1 and ch_in <= 4096 and\
        ch_out >= 1 and ch_out <= 4096 and\
        dilation_h in [1, 2, 4] and\
        dilation_w in [1, 2, 4]


@pyxir.register_op_support_check('DPUCADX8G', 'Conv2DTranspose')
def conv2d_transpose_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2DTranspose operator
        on the DPUCADX8G target """

    data_layout = X.attrs['data_layout']

    kernel_h, kernel_w = X.attrs['kernel_size']
    stride_h, stride_w = X.attrs['strides']
    dilation_h, dilation_w = X.attrs['dilation']
    padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
        X.attrs['padding'][data_layout.index('W')]
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w
    padding = X.attrs['padding']

    ch_in, ch_out = X.attrs['channels']
    groups = X.attrs['groups']

    # TODO padding?
    # padding_h_top >= 0 and padding_h_top <= kernel_h - 1 and\
    # padding_h_bot >= 0 and padding_h_bot <= kernel_h - 1 and\
    # padding_w_left >= 0 and padding_w_left <= kernel_w - 1 and\
    # padding_w_right >= 0 and padding_w_right <= kernel_w - 1 and\
    return groups == 1 and\
        kernel_h >= 1 and kernel_h <= 15 and\
        kernel_w >= 1 and kernel_w <= 15 and\
        stride_h in [1, 2, 4, 8] and\
        stride_w in [1, 2, 4, 8] and\
        ch_in >= 1 and ch_in <= 4096 and\
        ch_out >= 1 and ch_out <= 4096 and\
        dilation_h in [1, 2, 4] and\
        dilation_w in [1, 2, 4]


@pyxir.register_op_support_check('DPUCADX8G', 'DPU')
def dpuv2_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided DPU operator
        on the DPUCADX8G target """

    # TODO out_ch

    return True


@pyxir.register_op_support_check('DPUCADX8G', 'Eltwise')
def eltwise_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Eltwise operator
        on the DPUCADX8G target """

    # TODO in_ch

    return True


@pyxir.register_op_support_check('DPUCADX8G', 'Pad')
def pooling_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the DPUCADX8G target """

    padding = X.attrs['padding']

    # TODO: padding?
    # padding_h_top >= 0 and padding_h_top <= 4 and\
    # padding_h_bot >= 0 and padding_h_bot <= 4 and\
    # padding_w_left >= 0 and padding_w_left <= 4 and\
    # padding_w_right >= 0 and padding_w_right <= 4 and\
    return True


@pyxir.register_op_support_check('DPUCADX8G', 'Pooling')
def pooling_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the DPUCADX8G target """

    data_layout = X.attrs['data_layout']

    kernel_h, kernel_w = X.attrs['kernel_size']
    stride_h, stride_w = X.attrs['strides']
    padding_h, padding_w = X.attrs['padding'][data_layout.index('H')],\
        X.attrs['padding'][data_layout.index('H')]
    padding_h_top, padding_h_bot = padding_h
    padding_w_left, padding_w_right = padding_w

    channels = X.shapes[data_layout.index('C')]

    # TODO: padding?
    # padding_h_top >= 0 and padding_h_top <= 4 and\
    # padding_h_bot >= 0 and padding_h_bot <= 4 and\
    # padding_w_left >= 0 and padding_w_left <= 4 and\
    # padding_w_right >= 0 and padding_w_right <= 4 and\
    return kernel_h >= 1 and kernel_h <= 15 and\
        kernel_w >= 1 and kernel_w <= 15 and\
        stride_h in [1, 2, 4, 8] and\
        stride_w in [1, 2, 4, 8] and\
        channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADX8G', 'Mean')
def mean_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Mean operator
        on the DPUCADX8G target """

    axes = X.attrs['axes']
    keepdims = X.attrs['keepdims']

    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check('DPUCADX8G', 'pReLU')
def prelu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided pRelu operator
        on the DPUCADX8G target """

    # TODO: supported ??

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('DPUCADX8G', 'LeakyReLU')
def leaky_relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided pRelu operator
        on the DPUCADX8G target """

    # TODO: supported ??

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('DPUCADX8G', 'ReLU')
def relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the DPUCADX8G target """

    # TODO always?

    return True


# @pyxir.register_op_support_check('DPUCADX8G', 'ReLU6')
# def relu6_op_support(X, bXs, tXs):
#     # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
#     """ Check whether we can execute the provided ReLU operator
#         on the DPUCADX8G target """

#     # TODO always?

#     return True


@pyxir.register_op_support_check('DPUCADX8G', 'Scale')
def scale_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Scale operator
        on the DPUCADX8G target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return axis != -1 and channels > 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADX8G', 'Upsampling2D')
def upsampling2d_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Upsampling2D operator
        on the DPUCADX8G target """

    method = X.attrs['method']

    return method == 'nearest_neighbor'
