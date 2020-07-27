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

""" Module for registering global Transpose OpaqueFunc """

import numpy as np

from typing import List

from pyxir.type import TypeCode
from pyxir.shared.xbuffer import XBuffer
from pyxir.opaque_func import OpaqueFunc
from pyxir.opaque_func_registry import register_opaque_func


@register_opaque_func('px.globals.Transpose',
                      [TypeCode.vXBuffer, TypeCode.vXBuffer, TypeCode.vInt])
def transpose_opaque_func(in_tensors: List[XBuffer],
                          out_tensors: List[XBuffer],
                          axes: List[int]):
    """ Expose a global Transpose function """
    out_tensors[0].copy_from(np.transpose(in_tensors[0], axes=tuple(axes)))
