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

"""Module for Decent quantizer simulation runtime"""

import os
import numpy as np

from typing import List, Dict, Callable

from ...graph import XLayer
from ..base_runtime import BaseRuntime


class RuntimeDecentQSim(BaseRuntime):

    """Runtime for Decent quantizer simulation"""

    def __init__(self,
                 name,
                 network,
                 params,
                 device: str = 'cpu',
                 batch_size: int = -1,
                 placeholder: bool = False,
                 **kwargs):
        super(RuntimeDecentQSim, self).__init__(
            name, network, params, device, batch_size, placeholder)

        if 'quant_keys' not in kwargs:
            raise ValueError("Trying to simulate unquantized model. Make sure to first"\
                             " quantize the model.")
        qkey = kwargs['quant_keys'][0]
        self.q_eval = kwargs[qkey]['q_eval']
        self.gpu = 0

    def _init_net(self, network: List[XLayer], params: Dict[str, np.ndarray]):
        # Do nothing
        pass

    def run(self,
            inputs: Dict[str, np.ndarray],
            outputs: List[str] = [],
            stop: str = None,
            force_stepwise: bool = False,
            debug: bool = False) -> List[np.ndarray]:
        """Override run method"""
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        input_graph_def = tf.Graph().as_graph_def()
        input_graph_def.ParseFromString(tf.io.gfile.GFile(self.q_eval, "rb").read())
        tf.import_graph_def(input_graph_def, name='')

        # TODO: input name
        in_tensors = {k: tf.compat.v1.get_default_graph().get_tensor_by_name('xinput' + str(i) + ':0')
                      for i, k in enumerate(inputs)}
        feed_dict = {in_tensors[k]: v for k, v in inputs.items()}
        out_tensors = [tf.compat.v1.get_default_graph().get_tensor_by_name(o + "/aquant" + ":0") for o in outputs]

        with tf.compat.v1.Session() as sess:
            out = sess.run(out_tensors, feed_dict=feed_dict)
            return out if isinstance(out, list) else [out]
