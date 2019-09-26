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
Module for XLayer neural network layers implemented on top of tensorflow


"""

import os
import abc
import math
import numpy as np
import tensorflow as tf
import logging

from .. import rt_layer

logger = logging.getLogger("pyxir")


class RtLayerTF(rt_layer.RtLayer):

    __metaclass__ = abc.ABCMeta

    dtype_to_tf = {
        'float32': tf.float32,
        'int8': tf.int8,
        'int16': tf.int16,
        'int32': tf.int32,
        'int64': tf.int64
    }

    dtype_to_np = {
        'float32': np.float32,
        'int8': np.int8,
        'int16': np.int16,
        'int32': np.int32,
        'int64': np.int64
    }
