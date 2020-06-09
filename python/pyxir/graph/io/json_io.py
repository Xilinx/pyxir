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
Module for XGraph JSON encoder and decoder


"""

import json

from pyxir.shapes import TensorShape, TupleShape


class XGraphJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        """ Encode XLayer unserializable types """

        if isinstance(obj, (TensorShape, TupleShape)):
            return obj.tolist()

        return super(XGraphJSONEncoder, self).default(obj)
