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
Module for registering quantization simulation transformation functions


"""

QUANTIZE_LAYER = {}


def register_quant_sim_transform(xop_name):
    # type: (str) -> None
    """ Return decorator for registering function to transform an operation
        of provided type for quantization simulation """

    def __register_quant_sim_transform(transform_func):
        # type: (function) -> function
        QUANTIZE_LAYER[xop_name] = transform_func

        return transform_func

    return __register_quant_sim_transform
