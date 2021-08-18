# Copyright 2021 Xilinx Inc.
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
"""Utilities for testing TF layers"""

import unittest
import numpy as np

try:
    from pyxir.runtime.tensorflow.x_2_tf_registry import X_2_TF
except ModuleNotFoundError:
    raise unittest.SkipTest("Skipping Tensorflow related test because Tensorflow is not available")


def build_exec_layers(xlayers, input_shapes, params):
    exec_layers = []
    for X in xlayers:
        new_layers = X_2_TF[X.type[0]](X, input_shapes, params)
        input_shapes[X.name] = new_layers[-1].shape
        exec_layers.extend(new_layers)
    return exec_layers


def execute_layers(layers, inputs):
    for layer in layers:
        inpts = [inputs[name] for name in layer.inputs]
        out = layer.forward_exec(inpts)
        inputs[layer.name] = out
    return out