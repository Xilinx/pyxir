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
""""Module for common target components"""

from pyxir.shared.container import StrContainer
from pyxir.opaque_func_registry import OpaqueFuncRegistry


def is_dpuczdx8g_vart_flow_enabled():
    of = OpaqueFuncRegistry.Get("pyxir.use_dpuczdx8g_vart")
    s = StrContainer("")
    of(s)
    return s.get_str() == "True"
