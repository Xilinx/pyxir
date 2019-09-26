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


import libpyxir as lpx


def py_func(args):
    print("test")
    #print("args", args[0].xg.get_name())
    #args[0].xg.set_name("xg_new")


# Create opaque function
of = lpx.OpaqueFunc(py_func)

# Register OpaqueFunc
ofr = lpx.OpaqueFuncRegistry.Register("xg")
ofr.set_func(of)
