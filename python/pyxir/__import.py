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
Module for importing and setting up libraries


"""

import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# import cvx modules if not yet in path
if sys.version_info >= (3, 0):
    from importlib import util
    cvx_spec = util.find_spec("cvx")
else:
    import imp
    try:
        cvx_spec = imp.find_module("cvx")
    except Exception as e:
        cvx_spec = None

if cvx_spec is None:
    cvx_dir = os.path.join(FILE_DIR, "../../lib/cvx")
    if os.path.exists(cvx_dir):
        sys.path.append(cvx_dir)
        # raise ValueError("Could not find cvx package, please install cvx")
