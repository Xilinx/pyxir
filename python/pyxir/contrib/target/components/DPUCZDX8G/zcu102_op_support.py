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

logger = logging.getLogger('pyxir')


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'BatchNorm')
def batchnorm_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BatchNorm operator
        on the zcu102 target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'BiasAdd')
def biasadd_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BiasAdd operator
        on the zcu102 target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Cast')
def cast_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Cast operator
        on the DPUCZDX8G-zcu102 target """

    dtype = X.attrs['dtype']

    return dtype == 'float32'


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Concat')
def concat_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Concat operator
        on the zcu102 target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Convolution')
def conv2d_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2D operator
        on the zcu102 target """

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
        ch_in >= 1 and ch_in <= 4096 and\
        ch_out >= 1 and ch_out <= 4096 and\
        dilation_h * ch_in <= 4096 and\
        (dilation_h == 1 or stride_h == 1) and\
        dilation_w * ch_in <= 4096 and\
        (dilation_w == 1 or stride_w == 1)


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Conv2DTranspose')
def conv2d_transpose_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2DTranspose operator
        on the zcu102 target """

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
        stride_w * ch_out >= 1 and stride_w * ch_out <= 4096 and\
        stride_h >= 1 and\
        padding_h_top >= 0 and padding_h_top <= kernel_h - 1 and\
        padding_h_bot >= 0 and padding_h_bot <= kernel_h - 1 and\
        padding_w_left >= 0 and padding_w_left <= kernel_w - 1 and\
        padding_w_right >= 0 and padding_w_right <= kernel_w - 1 and\
        ch_in >= 1 and ch_in <= 4096 and\
        ch_out >= 1 and ch_out <= 4096 and\
        dilation_h * ch_in <= 4096 and\
        (dilation_h == 1 or stride_h == 1) and\
        dilation_w * ch_in <= 4096 and\
        (dilation_w == 1 or stride_w == 1)


# @pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Dense')
# def dense_op_support(X, bXs, tXs):
#     # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
#     """ Check whether we can execute the provided Dense operator
#         on the zcu102 target """

#     # TODO out_ch

#     return True


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'DPU')
def DPUCZDX8G_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided DPU operator
        on the zcu102 target """

    return True


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Eltwise')
def eltwise_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Eltwise operator
        on the zcu102 target """

    # TODO in_ch

    return True


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Pad')
def pad_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the zcu102 target """

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


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Pooling')
def pooling_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the zcu102 target """

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
        channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Mean')
def mean_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Mean operator
        on the zcu102 target """

    axes = X.attrs['axes']
    keepdims = X.attrs['keepdims']

    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'LeakyReLU')
def leaky_relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided LeakyRelu operator
        on the zcu102 target """

    # TODO: position?

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'pReLU')
def prelu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided pRelu operator
        on the zcu102 target """

    # TODO: position?

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'ReLU')
def relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the zcu102 target """

    # TODO always?

    return True


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'ReLU6')
def relu6_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the zcu102 target """

    # TODO always?

    return True


@pyxir.register_op_support_check('DPUCZDX8G-zcu102', 'Scale')
def scale_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Scale operator
        on the zcu102 target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels > 1 and channels <= 4096
