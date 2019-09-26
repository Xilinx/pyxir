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
Module for loading models using TVM


"""


def load_model_from_file(frontend, framework):
    # type: (str, str) -> Function
    """
    TODO
    """
    from . import relay_util

    load_funcs = {
        'Relay': {
            'Tensorflow': relay_util.from_tensorflow,
            'MXNet': relay_util.from_mxnet,
            'CoreML': relay_util.from_coreml,
            'Caffe': relay_util.from_caffe,
            'Caffe2': relay_util.from_caffe2,
            'ONNX': relay_util.from_onnx,
            'PyTorch': relay_util.from_pytorch,
            'Relay': relay_util.from_relay
        }
    }

    try:
        load_func = load_funcs[frontend][framework]
    except KeyError as e:
        raise ValueError("The frontend/framework combination: ({},{}) for"
                         " loading a model from file using TVM is invalid."
                         " Valid arguments are all combinations of [NNVM, "
                         " Relay] and [Tensorflow, CoreML, Caffe, Caffe2, "
                         " ONNX].")
    return load_func
