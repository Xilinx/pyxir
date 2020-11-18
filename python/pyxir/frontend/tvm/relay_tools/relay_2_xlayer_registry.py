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
Data structure for registering and tracking Relay to XLayer converters
"""

import logging

logger = logging.getLogger('pyxir')


class Relay2XLayerRegistry(object):

    class __Relay2XLayerRegistry(object):
        """ Implementation of Relay2XLayerRegistry Singleton """

        def __init__(self):
            self.relay_2_xlayer = {}

        def __getitem__(self, relay_op):
            """ Retrieve Relay to XLayer converter """
            if relay_op not in self.relay_2_xlayer:
                return self.relay_2_xlayer['RelayOp']
            return self.relay_2_xlayer[relay_op]

        def __setitem__(self, relay_op, relay_2_xlayer_func):
            """ Add Relay to XLayer converter function """
            if relay_op in self.relay_2_xlayer:
                raise ValueError("Can't register a Relay to XLayer conversion"
                                 " function for operation: {} because the"
                                 " operation has already been registered."
                                 .format(relay_op))

            self.relay_2_xlayer[relay_op] = relay_2_xlayer_func

    # Singleton storage
    __instance = None

    def __init__(self):
        """ Create singleton instance """

        # Check whether we already have an instance
        if Relay2XLayerRegistry.__instance is None:
            Relay2XLayerRegistry.__instance = \
                Relay2XLayerRegistry.__Relay2XLayerRegistry()

        self.__dict__['Relay2XLayerRegistry__instance'] = \
            Relay2XLayerRegistry.__instance

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


def register_relay_2_xlayer_converter(relay_op):
    # type: (str) -> Function
    """
    Return decorator for converting a relay operation to an XLayer
    """

    registry = Relay2XLayerRegistry()

    def register_relay_2_xlayer_converter_decorator(conversion_func):
        # type: (Function) -> None
        registry[relay_op] = conversion_func

    return register_relay_2_xlayer_converter_decorator


def register_relay_2_xlayer_converter_base(relay_op):
    # type: (str) -> Function
    """
    Return decorator for converting a relay operation to an XLayer
    """

    registry = Relay2XLayerRegistry()

    def base_relay_2_xlayer(specific_relay_2_xlayer):

        def __base_relay_2_xlayer(expr, params, schedule, net, op_idx,
                                  RELAY_2_XLAYER, **kwargs):
            # type: (tvm.relay.expr.Expr, Dict[str, numpy.ndarray], List[Expr],
            #   Dict[int, XLayer], Dict[str, int], Dict[str, Function])
            #   -> XLayer
            """ Base conversion function for one dynamic input layer
                expressions to avoid duplication. Example dynamic input
                layer expressions: Log, BatchNorm, Conv2D... Multiple
                dynamic input layer expressions: Add, Multiply, Concat... """

            if expr in net:
                # This expressions is already transformed so we reuse that one
                logger.debug("MEMORY: {}".format(relay_op))
                return net[expr]

            iXs = []
            for in_expr in expr.args:
                in_expr_class = in_expr.__class__.__name__
                iX = RELAY_2_XLAYER[in_expr_class](in_expr, params,
                                                   schedule, net, op_idx,
                                                   RELAY_2_XLAYER, **kwargs)
                iXs.append(iX)

            logger.debug('{}:'.format(relay_op))

            # Update schedule with child layers
            for in_expr, iX in zip(expr.args, iXs):
                if in_expr not in net:
                    schedule.append(in_expr)
                    net[in_expr] = iX

            # Create XLayer
            op_name = '{}-{}'.format(relay_op, str(hash(expr)))

            X = specific_relay_2_xlayer(op_name, expr, iXs)

            # !Important: set input layer tops
            for iX in iXs:
                iX.tops.append(op_name)

            return X

        return __base_relay_2_xlayer

    def register_relay_2_xlayer_converter_decorator(specific_relay_2_xlayer):
        # type: (Function) -> None
        registry[relay_op] = base_relay_2_xlayer(specific_relay_2_xlayer)

    return register_relay_2_xlayer_converter_decorator
