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
Frontend module for importing from TVM IRs


"""

import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check if we can import necessary tvm modules
if sys.version_info >= (3, 0):
    import importlib
    tvm_spec = importlib.util.find_spec("tvm")
    topi_spec = importlib.util.find_spec("topi")
    # nnvm_spec = importlib.util.find_spec("nnvm")
else:
    import imp
    try:
        tvm_spec = imp.find_module("tvm")
    except:
        tvm_spec = None
    try:
        topi_spec = imp.find_module("topi")
    except:
        topi_spec = None
    # try:
    #     nnvm_spec = imp.find_module("nnvm")
    # except:
    #     nnvm_spec = None

if tvm_spec is None:
    tvm_dir = os.path.join(FILE_DIR, "../../../../lib/tvm/python")
    if not os.path.exists(tvm_dir):
        raise ValueError("Could not find tvm package, please install before"
                         " using tvm functionality")
    sys.path.append(tvm_dir)
if topi_spec is None:
    topi_dir = os.path.join(FILE_DIR, "../../../../lib/tvm/topi/python")
    if not os.path.exists(topi_dir):
        raise ValueError("Could not find (tvm) topi package, please install"
                         " before using (tvm) topi functionality")
    sys.path.append(topi_dir)
# if nnvm_spec is None:
#     nnvm_dir = os.path.join(FILE_DIR, "../../../../lib/tvm/nnvm/python")
#     if not os.path.exists(nnvm_dir):
#         raise ValueError("Could not find (tvm) nnvm package, please install"
#                          " before using (tvm) nnvm functionality")
#     sys.path.append(nnvm_dir)

from .io import load_model_from_file
from .relay import from_relay
