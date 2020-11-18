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

""" Factory module for creating Runtimes """

import logging

from ..graph.layer import xlayer
from ..graph.xgraph_factory import XGraphFactory
from pyxir.runtime.base_runtime import BaseRuntime

logger = logging.getLogger('pyxir')


class RuntimeFactory(object):

    """
    The RuntimeFactory Singleton is responsible for creating
    BaseRuntime objects
    """

    class __RuntimeFactory(object):

        def __init__(self):
            self.xgraph_factory = XGraphFactory()
            self._runtimes = {}

        def _get_net_and_params(self, xgraph, last_layers):
            # type: (Xgraph, List[str]) -> List[XLayer], Dict[str, np.ndarray]
            """ Return the XGraph submodel as a list of XLayers and the
            parameters provided the given last layers of the runtime model
            """

            net = []
            params = {}
            last_layer_cnt = 1
            last_layer_tops = set([])

            for X in xgraph.get_layers():

                if X.name in last_layer_tops:
                    last_layer_tops = last_layer_tops.union(tuple(X.tops))
                    continue

                if 'Convolution' in X.type or 'Conv2DTranspose' in X.type:
                    if not isinstance(X.data, xlayer.ConvData):
                        raise ValueError(
                            "Invalid convolution data type: {}, should be "
                            " xlayer.ConvData".format(type(X.data)))
                    # OIHW
                    params[X.name + '_kernel'] = X.data.weights
                    params[X.name + '_biases'] = X.data.biases
                elif 'Dense' in X.type:
                    if not isinstance(X.data, xlayer.ConvData):
                        raise ValueError(
                            "Invalid inner product data type: {}, should be "
                            " xlayer.ConvData".format(type(X.data)))
                    # OIHW
                    params[X.name + '_weights'] = X.data.weights
                    params[X.name + '_biases'] = X.data.biases
                elif 'BatchNorm' in X.type:
                    if not isinstance(X.data, xlayer.BatchData):
                        raise ValueError(
                            "Invalid batchnorm data type: {}, should be"
                            " xlayer.BatchData".format(type(X.data)))
                    # channels
                    params[X.name + '_mu'] = X.data.mu
                    params[X.name + '_variance'] = X.data.sigma_square
                    params[X.name + '_gamma'] = X.data.gamma
                    params[X.name + '_beta'] = X.data.beta
                elif 'Scale' in X.type:
                    if not isinstance(X.data, xlayer.ScaleData):
                        raise ValueError(
                            "Invalid scale data type: {}, should be"
                            " xlayer.ScaleData".format(type(X.data)))
                    # channels
                    params[X.name + '_gamma'] = X.data.gamma
                    params[X.name + '_beta'] = X.data.beta
                elif 'BiasAdd' in X.type:
                    assert X.data is not None
                    params[X.name + '_bias'] = X.data[0]
                elif 'Eltwise' in X.type:
                    if X.data != []:
                        params[X.name + '_beta'] = X.data[0]

                net.append(X)

                if last_layers is not None and X.name in last_layers:
                    if last_layer_cnt == len(last_layers):
                        break
                    else:
                        last_layer_cnt += 1
                        last_layer_tops = last_layer_tops.union(tuple(X.tops))

            return net, params

        def build_runtime(self,
                          xgraph,
                          runtime='cpu-tf',
                          target='cpu',
                          last_layers=None,
                          batch_size=-1,
                          placeholder=False,
                          out_tensor_names=None,
                          **kwargs):
            # type: (str, XGraph, str, str, List[str], int) -> BaseRuntime
            """
            Build an runtime graph based on the given target (e.g. tensorflow)
            """

            net, params = self._get_net_and_params(xgraph, last_layers)

            logger.info("End building Runtime")
            logger.info("Layers: {}".format(len(net)))
            logger.debug([X.name for X in net])

            # input_names = xgraph.get_input_names()
            output_names = set(xgraph.get_output_names())
            hidden_out_tensor_names = [otn for otn in out_tensor_names if otn not in output_names]\
                if out_tensor_names is not None else []

            return self._runtimes[runtime](xgraph.get_name(), net, params, target, batch_size,
                                           placeholder, hidden_out_tensor_names=hidden_out_tensor_names,
                                           **kwargs)

        def register_exec_graph(self, rt_name: str, runtime: BaseRuntime):
            """Register a creator for a new Runtime subclass"""
            if rt_name in self._runtimes:
                raise ValueError("This runtime is already registered")
            if not issubclass(runtime, BaseRuntime):
                raise ValueError("Provided runtime should be a"
                                 " subclass of Runtime")

            self._runtimes[rt_name] = runtime

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """Create singleton instance"""
        # Check whether we already have an instance
        if RuntimeFactory.__instance is None:
            # Create and remember instance
            RuntimeFactory.__instance = \
                RuntimeFactory.__RuntimeFactory()

        # Store instance reference as the only member in the handle
        self.__dict__['_RuntimeFactory__instance'] = \
            RuntimeFactory.__instance

    def __getattr__(self, attr):
        """Delegate access to implementation"""
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """Delegate access to implementation"""
        return setattr(self.__instance, attr, value)
