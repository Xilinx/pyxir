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
else:
    import imp
    try:
        tvm_spec = imp.find_module("tvm")
    except:
        tvm_spec = None

if tvm_spec is None:
    tvm_dir = os.path.join(FILE_DIR, "../../../../lib/tvm/python")
    if not os.path.exists(tvm_dir):
        raise ValueError("Could not find tvm package, please install before"
                         " using tvm functionality")
    sys.path.append(tvm_dir)

from .io import load_model_from_file
from .relay import from_relay
