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
Data structure for registering and tracking ONNX to XLayer conversion
functions


"""

import logging

from pyxir.graph.layer import xlayer_factory as xlf

logger = logging.getLogger('pyxir')


class ONNX2XLayerRegistry(object):

    class __ONNX2XLayerRegistry(object):
        """ Implementation of ONNX2XLayerRegistry Singleton """

        def __init__(self):
            self.onnx_2_xlayer = {}

        def __getitem__(self, onnx_op):
            """ Retrieve ONNX to XLayer converter """
            if onnx_op not in self.onnx_2_xlayer:
                raise NotImplementedError("ONNX op: {} to XLayer conversion"
                                          " not implemented yet"
                                          .format(onnx_op))
                # return self.onnx_2_xlayer['ONNXOp']
            return self.onnx_2_xlayer[onnx_op]

        def __setitem__(self, onnx_op, onnx_2_xlayer_func):
            """ Add ONNX to XLayer converter function """
            if onnx_op in self.onnx_2_xlayer:
                raise ValueError("Can't register a ONNX to XLayer conversion"
                                 " function for operation: {} because the"
                                 " operation has already been registered."
                                 .format(onnx_op))

            self.onnx_2_xlayer[onnx_op] = onnx_2_xlayer_func

        def __contains__(self, onnx_op):
            """ Return whether this registry conatins the provided op """
            return onnx_op in self.onnx_2_xlayer

        def remove(self, onnx_op):
            """ Delete the provided onnx op from the registry """
            del self.onnx_2_xlayer[onnx_op]

    # Singleton storage
    __instance = None

    def __init__(self):
        """ Create singleton instance """

        # Check whether we already have an instance
        if ONNX2XLayerRegistry.__instance is None:
            ONNX2XLayerRegistry.__instance = \
                ONNX2XLayerRegistry.__ONNX2XLayerRegistry()

        self.__dict__['ONNX2XLayerRegistry__instance'] = \
            ONNX2XLayerRegistry.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)

    def __getitem__(self, attr):
        """ Delegate access to implementation """
        return self.__instance[attr]

    def __setitem__(self, attr, value):
        """ Delegate access to implementation """
        self.__instance[attr] = value

    def __contains__(self, attr):
        """ Delegate access to implementation """
        return attr in self.__instance


def register_onnx_2_xlayer_converter(onnx_op_type):
    # type: (str) -> Function
    """ Return a decorator for registering an ONNX to XLayer
        conversion function """

    registry = ONNX2XLayerRegistry()

    def base_onnx_2_xlayer(onnx_2_xlayer_func):

        def __base_onnx_2_xlayer(node, params, xmap):
            # type: (NodeWrapper, Dict[str, np.ndarray], Dict[str, XLayer])
            #   -> List[XLayer]
            """ Base ONNX to XLayer conversion function """

            bottoms = node.get_inputs()
            bXs = [xmap[b] for b in bottoms]

            Xs = onnx_2_xlayer_func(node, params, xmap)

            for X in Xs:
                # xmap[X.name] = X
                xmap[X.attrs['onnx_id']] = X

            # TODO: necessary for partitioning, but make more robust
            for bX in bXs:
                bX.tops.append(Xs[0].name)

            return Xs

        return __base_onnx_2_xlayer

    def register_onnx_2_xlayer_converter_decorator(onnx_2_xlayer_func):
        # type: (Function) -> None
        registry[onnx_op_type] = base_onnx_2_xlayer(onnx_2_xlayer_func)

        return onnx_2_xlayer_func

    return register_onnx_2_xlayer_converter_decorator
