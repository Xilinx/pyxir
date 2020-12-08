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

"""Tensorflow runtime module"""

import copy
import warnings
import numpy as np
import tensorflow as tf
import logging

from typing import List

from pyxir.shared import fancy_logging

from .. import base
from ..base_runtime import BaseRuntime
from .x_2_tf_registry import X_2_TF

logger = logging.getLogger("pyxir")
fancy_logger = fancy_logging.getLogger("pyxir")


class RuntimeTF(BaseRuntime):
    """Runtime on top of Tensorflow"""

    def __init__(self,
                 name: str,
                 network: list,
                 params: dict,
                 device: str = 'cpu',
                 batch_size: int = -1,
                 placeholder: bool = False,
                 hidden_out_tensor_names: List[str] = None,
                 **kwargs):
        tf.compat.v1.reset_default_graph()

        self.tf_step_graph = tf.Graph()
        with self.tf_step_graph.as_default() as g:
            with g.name_scope('tf_step_graph'):
                super(RuntimeTF, self).__init__(
                    name, network, params, device, batch_size, placeholder)
        
        # In the Tensorflow model we might need to create a dummy output layer
        # if an intermediate layer is an out as in for example: Relu -> Conv
        #                                                           \-> Conv
        # if the Relu is also an output of the model. This is required for 
        # some of the quantization/compilation tools to parse the outputs 
        # correctly
        self.hidden_out_tensor_names = hidden_out_tensor_names \
            if hidden_out_tensor_names is not None else []

        # TODO Explanation
        self.compiler_target = kwargs['compiler_target'] if 'compiler_target' in kwargs else None

        # The Runtime builds an executable graph as a list of executable layers
        # To execute the graph on a given input we have to call forward execute
        # layer by layer. This is very inefficient in tensorflow due to the
        # overhead of starting a session run for each layer. Therefore, we
        # build a separate tensorflow graph that is more efficient
        tf.compat.v1.reset_default_graph()
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default() as g:
            self.init_tf_graph(self.net)

    def init_tf_graph(self, net):
        # type: (List[RtLayer]) -> None
        fancy_logger.banner("INIT TF NET")

        self.tf_inputs = set()
        self.tf_tensors = {}
        # All outputs of the graph, used for optimization
        self.tf_outputs = []

        for idx, layer in enumerate(net):

            logger.info("-----------------------")
            logger.info("Layer idx: {}, layer name: {}"
                        .format(idx, layer.name))
            logger.info(f"Layer type: {layer.type}")
            logger.info("Layer inputs: {}".format(layer.get_input_names()))

            # TODO Explanation
            in_tensors = []
            for inpt in layer.get_input_names():
                if inpt in self.hidden_out_tensor_names:
                    in_tensors.append(self.tf_tensors[inpt + "_hidden"])
                elif inpt in self.tf_tensors:
                    in_tensors.append(self.tf_tensors[inpt])
            
            layer_name = layer.name
            if layer_name not in self.hidden_out_tensor_names:
                out_tensors = layer.get_output_tensors(in_tensors, compiler_target=self.compiler_target)
            else:
                out_tensors = layer.get_output_tensors(in_tensors,
                                                       override_name=layer.name + "_hidden",
                                                       compiler_target=self.compiler_target)
                layer_name = layer.name + "_hidden"

            logger.info("Out shapes: {}"
                        .format([o.shape if not isinstance(o, list)
                                 else [oo.shape for oo in o]
                                 for o in out_tensors]))

            self.tf_tensors[layer_name] = out_tensors[0]
            # For now, the first output is the main outputs,
            #   other outputs are returned for optimization purposes
            self.tf_outputs.extend(out_tensors[1:])
            if layer.is_input_layer():
                self.tf_inputs.add(layer_name)

            # TODO
            if layer.name in self.hidden_out_tensor_names:
                inpt = self.tf_tensors[layer_name]
                if self.compiler_target == 'DPUv1Compiler':
                    # identity = tf.add(
                    #     tf.multiply(inpt, np.ones((1,), dtype=np.float32), name=layer.name),
                    #     np.zeros((1,), dtype=np.float32),
                    #     name=layer.name + "/Add"
                    # )
                    # TODO
                    channels = inpt.shape[-1]
                    kernel = tf.constant(np.ones((1, 1, channels, channels), dtype=np.float32))
                    identity = tf.nn.conv2d(inpt, kernel, strides=[1], padding='VALID', name=layer.name)
                else:
                    identity = tf.add(
                        tf.multiply(inpt, np.ones((1,), dtype=np.float32)),
                        np.zeros((1,), dtype=np.float32),
                        name=layer.name
                    )
                self.tf_tensors[layer.name] = identity

        # TODO: Merge with tf_outputs?
        logger.info("-----------------------")

        self.tf_res = self.tf_tensors[net[-1].name]
        self.tf_outputs.append(self.tf_res)

        logger.debug(self.tf_res)
        logger.info("-----------------------")

    def _xfdnn_op_to_exec_op(self, op_type):
        # type: (str) -> function
        """
        Overwrites Runtime abstract method.

        Takes a operation type and returns a function of type:
        (XLayer, Dict[str,List[int]], Dict[str,numpy.ndarray],
            Dict[str,Dict]) -> List[rt_layer.RtLayer]
        that takes in a parameters layer object, inputs shapes dict, params
        dict and quantization parameters dict and outputs and returns a list
        of executable RtLayerTF objects
        """
        if op_type not in X_2_TF:
            raise NotImplementedError("Operation of type: {} is not supported"
                                      " on RuntimeTF"
                                      .format(op_type))
        return X_2_TF[op_type]

    def run_stepwise(self, inputs, stop=None):
        # (dict, str, str) -> (int, str, dict, numpy.ndarray, numpy.ndarray)
        """
        TODO
        """
        # Wrap stepwise execution with to make sure default graph is set
        with self.tf_step_graph.as_default():

            for idx, layer, inpts, outpt, quant_output \
                    in super(RuntimeTF, self).run_stepwise(inputs, stop):
                yield (idx, layer, inpts, outpt, quant_output)

        tf.compat.v1.reset_default_graph()

    def run(self, inputs, outputs=[], stop=None,
            force_stepwise=False, debug=False):
        # (Dict[str, numpy.ndarray], List[str], str, bool) ->
        #   List[numpy.ndarray]/numpy.ndarray
        """
        Overwrites the Runtime run function to use the efficient
        tensorflow graph to execute this computational graph on the
        given inputs

        inputs: Dict[str, numpy.ndarray]
            the inputs for this executable computational graph
        outputs: List[str]
            the output(s) to be returned
        stop: str
            the operation at which to stop running
        force_stepwise: bool (default: False)
            whether to force a stepwise calculation of the computational graph
            on the provided inputs (used for debugging purposes)

        Returns
        -------
        res: List[numpy.ndarray]/numpy.ndarray
            a list of outputs if multiple outputs were requested otherwise
            one output
        """
        fancy_logger.banner("RUN TF NET")

        # test = False
        if force_stepwise:
            # Use tensorflow stepwise graph because we have a stepwise
            #   tensorflow graph and a full tensorflow graph
            with self.tf_step_graph.as_default():
                return super(RuntimeTF, self).run(inputs, outputs, stop)
            tf.compat.v1.reset_default_graph()
        else:
            # inputs.update(self.params)
            if stop is not None:
                logger.warn("[WARNING] a stop operation was provided but when"
                            " running the efficient implementation, the"
                            " computations can't be stopped in the middle"
                            " of the graph")

            if not all([inpt in self.tf_inputs for inpt in inputs]):
                raise ValueError("Unknown inputs. The provided inputs are: {}."
                                 " \n The expected inputs are: {}. \n Provided"
                                 " but not expected: {}"
                                 .format(inputs.keys(), self.tf_inputs,
                                         [inpt for inpt in inputs if
                                          inpt not in self.tf_inputs]))
            if not all([inpt in inputs for inpt in self.tf_inputs]):
                raise ValueError("Missing inputs. The provided inputs are: {},"
                                 " but the expected inputs are: {}"
                                 .format(inputs.keys(), self.tf_inputs))
            if not all([outpt in self.tf_tensors for outpt in outputs]):
                raise ValueError("Unknown outputs. The provided outputs are:"
                                 " {}, but the possible outputs are: {}"
                                 .format(outputs, self.tf_tensors.keys()))

            tf_inputs = {self.tf_tensors[inpt]: inputs[inpt]
                         for inpt in inputs.keys()}
            tf_outputs = [self.tf_tensors[outpt] for outpt in outputs] if\
                len(outputs) > 0 else self.tf_res

            # logger.debug('tf_outputs')
            # logger.debug(tf_outputs)

            # TODO
            # tf_outputs.extend(self.tf_outputs)

            with tf.compat.v1.Session(graph=self.tf_graph) as sess:
                if debug:
                    writer = tf.summary.FileWriter("output", sess.graph)

                # logger.debug(tf_inputs)
                sess.run(tf.compat.v1.global_variables_initializer())
                out = sess.run(tf_outputs, feed_dict=tf_inputs)
                # logger.debug(out)
                # out = out[-1]

                if debug:
                    writer.close()

                return out if isinstance(out, list) else [out]

    def optimize(self, inputs, debug=False):
        # (Dict[str, numpy.ndarray], bool) ->
        #    Tuple(dict, List[numpy.ndarray]/numpy.ndarray)
        """
        Overwrites the Runtime optimize function to use the efficient
        tensorflow graph to execute this computational graph on the
        given inputs and return outputs AND all variables

        Arguments
        ---------
        inputs: Dict[str, numpy.ndarray]
            the inputs for this executable computational graph

        Returns
        -------
        res: List[numpy.ndarray]/numpy.ndarray
            a list of outputs if multiple outputs were requested otherwise
            one output
        vrs: Dict[str, numpy.ndarray]
            a dictionary from variable names to their optimized values
        """
        fancy_logger.banner("OPTIMIZE TF NET")

        # inputs.update(self.params)

        if not all([inpt in self.tf_inputs for inpt in inputs]):
            raise ValueError("Unknown inputs. The provided inputs are: {}. \n"
                             " The expected inputs are: {}. \n Provided but"
                             " not expected: {}"
                             .format(inputs.keys(), self.tf_inputs,
                                     [inpt for inpt in inputs
                                      if inpt not in self.tf_inputs]))
        if not all([inpt in inputs for inpt in self.tf_inputs]):
            raise ValueError("Missing inputs. The provided inputs are: {}, but"
                             " the expected inputs are: {}"
                             .format(inputs.keys(), self.tf_inputs))

        tf_inputs = {self.tf_tensors[inpt]: inputs[inpt]
                     for inpt in inputs.keys()}
        # self.tf_res
        tf_outputs = self.tf_outputs

        with tf.compat.v1.Session(graph=self.tf_graph) as sess:
            if debug:
                writer = tf.summary.FileWriter("output", sess.graph)

            sess.run(tf.compat.v1.global_variables_initializer())
            outs = sess.run(tf_outputs, feed_dict=tf_inputs)

            tvars = tf.compat.v1.trainable_variables()

            # logger.debug(tvars)
            tvars_vals = sess.run(tvars)
            # logger.debug(tvars_vals)

            # Note: strip :0 from tensorflow Variable
            vrs_dict = {
                tvar.name.split(':')[0]: value for tvar, value in
                zip(tvars, tvars_vals)
            }

            if debug:
                writer.close()

            return outs, vrs_dict
