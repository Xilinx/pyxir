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


@pyxir.register_op_support_check('DPUCADF8H', 'BatchNorm')
def batchnorm_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BatchNorm operator
        on the DPUCADF8H target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADF8H', 'BiasAdd')
def biasadd_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided BiasAdd operator
        on the DPUCADF8H target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADF8H', 'Cast')
def cast_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Cast operator
        on the DPUCADF8H target """

    dtype = X.attrs['dtype']

    return dtype == 'float32'

@pyxir.register_op_support_check('DPUCADF8H', 'Concat')
def concat_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Concat operator
        on the DPUCADF8H target """

    axis = X.attrs['axis']
    channels = X.shapes[axis]

    return channels >= 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADF8H', 'Convolution')
def conv2d_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2D operator
        on the DPUCADF8H target """

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


@pyxir.register_op_support_check('DPUCADF8H', 'Conv2DTranspose')
def conv2d_transpose_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Conv2DTranspose operator
        on the DPUCADF8H target """

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


@pyxir.register_op_support_check('DPUCADF8H', 'DPU')
def dpuv2_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided DPU operator
        on the DPUCADF8H target """

    # TODO out_ch

    return True


@pyxir.register_op_support_check('DPUCADF8H', 'Eltwise')
def eltwise_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Eltwise operator
        on the DPUCADF8H target """

    # TODO in_ch

    return True


@pyxir.register_op_support_check('DPUCADF8H', 'Maximum')
def maximum_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """Check whether we can execute the provided Maximum operator
        on the DPUCADF8H target

    Return true if part of leaky relu pattern    
    """
    # check whether part of leaky relu
    return 'patterns' in X.attrs and 'LeakyReLU' in X.attrs['patterns']


@pyxir.register_op_support_check('DPUCADF8H', 'Pad')
def pooling_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the DPUCADF8H target """

    padding = X.attrs['padding']

    # TODO: padding?
    # padding_h_top >= 0 and padding_h_top <= 4 and\
    # padding_h_bot >= 0 and padding_h_bot <= 4 and\
    # padding_w_left >= 0 and padding_w_left <= 4 and\
    # padding_w_right >= 0 and padding_w_right <= 4 and\
    return True


@pyxir.register_op_support_check('DPUCADF8H', 'Pooling')
def pooling_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Pooling operator
        on the DPUCADF8H target """

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


@pyxir.register_op_support_check('DPUCADF8H', 'Mean')
def mean_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Mean operator
        on the DPUCADF8H target """

    axes = X.attrs['axes']
    keepdims = X.attrs['keepdims']

    return len(axes) == 2 and keepdims


@pyxir.register_op_support_check('DPUCADF8H', 'pReLU')
def prelu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided pRelu operator
        on the DPUCADF8H target """

    # TODO: supported ??

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('DPUCADF8H', 'LeakyReLU')
def leaky_relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided pRelu operator
        on the DPUCADF8H target """

    # TODO: supported ??

    # Only LeakyReLU: alpha == 0.1 supported
    alpha = X.attrs['alpha']

    return math.isclose(alpha, 0.1, rel_tol=1e-5)


@pyxir.register_op_support_check('DPUCADF8H', 'ReLU')
def relu_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided ReLU operator
        on the DPUCADF8H target """

    # TODO always?

    return True


# @pyxir.register_op_support_check('DPUCADF8H', 'ReLU6')
# def relu6_op_support(X, bXs, tXs):
#     # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
#     """ Check whether we can execute the provided ReLU operator
#         on the DPUCADF8H target """

#     # TODO always?

#     return True


#@pyxir.register_op_support_check('DPUCADF8H', 'Scale')
#def scale_op_support(X, bXs, tXs):
#    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
#    """ Check whether we can execute the provided Scale operator
#        on the DPUCADF8H target """
#
#    axis = X.attrs['axis']
#    channels = X.shapes[axis]
#    # axis != -1 and 
#    return channels > 1 and channels <= 4096


@pyxir.register_op_support_check('DPUCADF8H', 'Upsampling2D')
def upsampling2d_op_support(X, bXs, tXs):
    # Type: (XLayer, List[XLayer], List[XLayer]) -> boolean
    """ Check whether we can execute the provided Upsampling2D operator
        on the DPUCADF8H target """

    method = X.attrs['method']

    return method == 'nearest_neighbor'
