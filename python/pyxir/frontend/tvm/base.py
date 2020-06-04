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
Generic module for converting external graph representations XGraph
representation


"""

import copy
import logging

from pyxir.graph.xgraph import XGraph
from pyxir.graph.xgraph_factory import XGraphFactory

logger = logging.getLogger("pyxir")


class BaseConverter(object):

    # __metaclass__ = abc.ABCMeta

    """
    Base class for converting external graph representations
    to xgraph representation
    """

    def __init__(self, output_png=None):
        self.output_png = output_png
        self.xgraph_factory = XGraphFactory()
