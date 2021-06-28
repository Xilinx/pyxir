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

"""Module for executing XGraphs"""
import sys
import os
import logging

from .rt_manager import RtManager
from .runtime_factory import RuntimeFactory
from .globals import transpose

logger = logging.getLogger("pyxir")
rt_manager = RtManager()
runtime_factory = RuntimeFactory()


try:
    # NOTE use RTLD_DEEPBIND dlopen flag to make sure TF uses it's own version of protobuf 
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    import tensorflow as tf
    sys.setdlopenflags(flags)
    # Register if we can import tensorflow
    from .tensorflow.runtime_tf import RuntimeTF, X_2_TF
    rt_manager.register_rt('cpu-tf', RuntimeTF, X_2_TF)
except Exception as e:
    logger.info("Could not load `cpu-tf` runtime because of error: {0}".format(e))

try:
    # Register if we can import numpy
    from .numpy.runtime_np import RuntimeNP, X_2_NP
    rt_manager.register_rt('cpu-np', RuntimeNP, X_2_NP)
except Exception as e:
    logger.info("Could not load `cpu-np` runtime because of error: {0}".format(e))

try:
    from .decentq_sim.runtime_decentq_sim import RuntimeDecentQSim
    rt_manager.register_rt('decentq-sim', RuntimeDecentQSim, {})
except Exception as e:
    logger.info("Could not load `decentq-sim` runtime because of error: {0}".format(e))
