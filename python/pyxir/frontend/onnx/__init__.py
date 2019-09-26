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
Frontend module for importing from ONNX models


"""

try:
    import onnx
except ImportError:
    raise ImportError("Please install ONNX (v1.5.0) before importing"
                      " ONNX models")

from .base import from_onnx
from .onnx_io import load_onnx_model_from_file
from .base import prequantize_onnx_model
