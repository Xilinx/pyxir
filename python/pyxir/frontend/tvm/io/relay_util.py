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
Utility module for loading models into the TVM Relay intermediate
representation (IR)


"""

import json
import logging

import tvm
import tvm.relay as relay

# TVM tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

logger = logging.getLogger('pyxir')


def from_tensorflow(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load tensorflow model from file and convert to relay expression
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("Please install tensorflow before trying to import"
                          " tensorflow models.")

    if outputs is None or len(outputs) < 1:
        raise ValueError("Please provide the output names for the provided"
                         " Tensorflow model")

    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

    mod, params = relay.frontend.from_tensorflow(
        graph_def,
        layout=None,
        shape=shapes,
        outputs=outputs
    )

    # data_layout = 'NHWC'
    # TODO: Get the right layout??
    return mod, params


def from_mxnet(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load MXNet model from file and convert to relay expression
    """
    try:
        import mxnet as mx
        from mxnet import gluon
    except ImportError as e:
        raise ImportError("Please install MXNet before trying to import"
                          " MXNet models. See https://mxnet.incubator.apache."
                          "org/get_started?version=v1.5.1&platform=linux&"
                          "language=python&environ=pip&processor=cpu for"
                          " installation instructions.")

    model = gluon.nn.SymbolBlock.imports(
        model_path,
        ['data'],  # TODO
        opt_model_path
    )
    mod, params = relay.frontend.from_mxnet(model, shapes)
    # model(mx.nd.ones((1,3,224,224)))

    # data_layout = 'NCHW'
    # TODO: Get the right layout??
    return mod, params


def from_coreml(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load CoreML model from file and convert to relay expression
    """
    raise NotImplementedError("")


def from_caffe(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load Caffe model from file and convert to relay expression
    """
    raise NotImplementedError("")


def from_caffe2(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load Caffe2 model from file and convert to relay expression
    """
    raise NotImplementedError("")


def from_onnx(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load ONNX model from file and convert to relay expression
    """
    try:
        import onnx
    except ImportError as e:
        raise ImportError("Please install onnx before trying to import"
                          " ONNX models.")
    from onnx import helper, shape_inference

    logger.debug("Model path: {}".format(model_path))
    onnx_model = onnx.load_model(model_path)
    onnx_graph = onnx_model.graph
    # for node in onnx_graph.node:
    #     print(node)
    onnx.checker.check_model(onnx_model)

    # inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)

    mod, params = relay.frontend.from_onnx(onnx_model, shape=shapes)

    return mod, params


def from_pytorch(model_path, shapes, outputs=None, opt_model_path=None,
                 model_class=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load PyTorch model from file and convert to relay expression
        using trace
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("Please install 'torch' before trying to import"
                          " PyTorch models.")

    if len(shapes) < 1:
        raise ValueError("Shape of input tensor should be provided in"
                         " shapes (dictionary) argument but got: {}"
                         .format(shapes))
    if len(shapes) > 1:
        raise ValueError("PyTorch frontend only supports models with"
                         " one input because of torch.jit.trace(...)"
                         " but got: {}  elements in shapes argument"
                         .format(len(shapes)))

    logger.debug("Model path: {}".format(model_path))

    if isinstance(model_class, torch.nn.Module):
        model = model_class()
    elif isinstance(opt_model_path, str):
        try:
            import torchvision.models
        except ImportError as e:
            raise ImportError("Please install 'torchvision' before trying"
                              " to import TorchVision models.")
        model_class = getattr(torchvision.models, opt_model_path)
        model = model_class()
    else:
        raise ValueError("Please provide a torch nn.Module subclass"
                         "to instantiate the model.")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    input_shape = [e if e != -1 else 1 for e in list(shapes.values())[0]]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    mod, params = relay.frontend.from_pytorch(scripted_model,
                                              shapes)

    return mod, params


def from_relay(model_path, shapes, outputs=None, opt_model_path=None):
    # type: (str, dict, List[str, ]str) -> relay.expr.Module, dict
    """ Load Relay model from file """

    with open(model_path, 'rb') as f:
        # json_str = json.load(f)
        mod = tvm.ir.load_json(f.read())

    with open(opt_model_path, "rb") as f:
        params = relay.load_param_dict(bytearray(f.read()))

    return mod, params
