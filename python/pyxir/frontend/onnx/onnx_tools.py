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
Utility module for parsing ONNX models


"""

import numpy as np

from onnx.numpy_helper import to_array

from typing import Any, Dict


def get_onnx_elem_type_2_dtype():
    return {
        1: 'float32',
        2: 'int8',
        3: 'uint8',
        4: 'uint16',
        5: 'int16',
        6: 'int32',
        7: 'int64',
        8: 'string',
        9: 'bool',
        10: 'float16',
        11: 'double',
        12: 'uint32',
        13: 'uint64',
        14: 'complex64',
        15: 'complex128',
        16: 'bfloat16'
    }


class NodeWrapper(object):

    attr_types = {
        0: 'UNDEFINED',
        1: 'FLOAT',
        2: 'INT',
        3: 'STRING',
        4: 'TENSOR',
        5: 'GRAPH',
        11: 'SPARSE_TENSOR',
        6: 'FLOATS',
        7: 'INTS',
        8: 'STRINGS',
        9: 'TENSORS',
        10: 'GRAPHS',
        12: 'SPARSE_TENSORS'
    }

    attr_types_inv = {
        'UNDEFINED': 0,
        'FLOAT': 1,
        'INT': 2,
        'STRING': 3,
        'TENSOR': 4,
        'GRAPH': 5,
        'SPARSE_TENSOR': 11,
        'FLOATS': 6,
        'INTS': 7,
        'STRINGS': 8,
        'TENSORS': 9,
        'GRAPHS': 10,
        'SPARSE_TENSORS': 12
    }

    def __init__(self, node):
        # type: (onnx.onnx_ONNX_RELEASE_ml_pb2.NodeProto) -> NodeWrapper
        self.node = node

    def get_op_type(self):
        return self.node.op_type

    def get_outputs(self):
        return list(self.node.output)

    def get_inputs(self):
        return list(self.node.input)

    def get_attributes(self) -> Dict[str, Any]:
        # type: () -> Dict[str, ?]
        """ Parses the ONNX node attributes and returns them
            in a dictionary """
        attrs = {}
        for attr in self.node.attribute:

            attr_name = attr.name
            attr_type = NodeWrapper.attr_types[attr.type]

            if attr_type == 'INT':
                attrs[attr_name] = int(attr.i)
            elif attr_type == 'INTS':
                attrs[attr_name] = [int(i) for i in attr.ints]
            elif attr_type == 'FLOAT':
                attrs[attr_name] = float(attr.f)
            elif attr_type == 'FLOATS':
                attrs[attr_name] = [float(f) for f in attr.floats]
            elif attr_type == 'STRING':
                attrs[attr_name] = attr.s.decode()
            elif attr_type == 'STRINGS':
                attrs[attr_name] = [s.decode() for s in attr.strings]
            elif attr_type == 'TENSOR':
                attrs[attr_name] = to_array(attr.t)
            elif attr_type == 'TENSORS':
                attrs[attr_name] = [to_array(t) for t in attr.tensors]
            else:
                raise NotImplementedError("Provided attribute type: {} is"
                                          " not yet supported for ONNX to"
                                          " XGraph conversion"
                                          .format(attr_type))

        return attrs

    def add_attribute(self, key: str, value: Any, attr_type: str):
        """ Add an attribute to the ONNX node """
        attr = self.node.attribute.add()
        attr.name = key
        attr.type = NodeWrapper.attr_types_inv[attr_type]

        if attr_type == "INT":
            attr.i = value
        elif attr_type == 'INTS':
            attr.ints.extend(value)
        elif attr_type == 'FLOAT':
            attr.f = value
        elif attr_type == 'FLOATS':
            attr.floatw.extend(value)
        elif attr_type == 'STRING':
            attr.s = value.encode()
        elif attr_type == 'STRINGS':
            attr.strings.extend([e.encode() for e in value])
        else:
            raise NotImplementedError("Provided attribute type: {} is"
                                      " not yet supported for ONNX to"
                                      " attribute "
                                      .format(attr_type))


class TensorTypeWrapper(object):

    """ Wrapper around ONNX TensorType """

    onnx_elem_type_2_dtype = get_onnx_elem_type_2_dtype()

    def __init__(self, tensor_type):
        self.tensor_type = tensor_type

    def get_dtype(self):
        return TensorTypeWrapper.onnx_elem_type_2_dtype[
            self.tensor_type.elem_type]

    def get_shape(self):
        return [int(i.dim_value) for i in self.tensor_type.shape.dim]
