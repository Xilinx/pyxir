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
Module for loading ONNX models from file
"""

import os
import onnx
import logging

from onnx import helper, shape_inference

logger = logging.getLogger('pyxir')


def load_onnx_model_from_file(model_path: str):
    """ Load ONNX model from file """

    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    logger.debug("Model path: {}".format(model_path))
    onnx_model = onnx.load_model(model_path)

    # TODO Should we check model?
    # onnx.checker.check_model(onnx_model)
    # inferred_model = shape_inference.infer_shapes(onnx_model)

    return onnx_model
