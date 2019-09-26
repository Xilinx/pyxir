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
Module for Pyxir IO APIs


"""

import os
import json

from pyxir.graph.xgraph import XGraph
from pyxir.graph.io.xgraph_io import XGraphIO
from pyxir.opaque_func_registry import register_opaque_func
from pyxir.type import TypeCode


def visualize(xgraph, pngfile='xgraph.png'):
    # type: (XGraph, str) -> None
    xgraph.visualize(pngfile)


def save(xgraph, filename):
    # type: (str) -> None
    """
    Save this XGraph to disk. The network graph information is written to
    json and the network paraemeters are written to an h5 file

    Arguments
    ---------
    xgraph: XGraph
        the XGraph to be saved
    filename: str
        the name of the files storing the graph inormation and network
        parameters
        the graph information is stored in `filename`.json
        the network paraemeters are stored in `filename`.h5
    """
    XGraphIO.save(xgraph, filename)


def load(net_file, params_file):
    # type: (str, str) -> XGraph
    """
    Load the graph network information and weights from the json network file
    respectively h5 parameters file

    Arguments
    ---------
    net_file: str
        the path to the file containing the network graph information
    params_file: str
        the path to the file containing the network weights
    """
    xgraph = XGraphIO.load(net_file, params_file)

    return xgraph


@register_opaque_func('pyxir.io.load_scheduled_xgraph_from_meta',
                      [TypeCode.Str, TypeCode.XGraph])
def load_scheduled_xgraph_opaque_func(build_dir: str,
                                      cb_scheduled_xgraph: XGraph):
    """
    Expose the load scheduled xgraph function as an opaque function
    so it can be called in a language agnostic way

    Arguments
    ---------
    build_dir: str
        the path to the build directory containing a meta.json file
    cb_scheduled_xgraph: XGraph
        return the scheduled XGraph
    """
    meta_file = os.path.join(build_dir, 'meta.json')

    if (not os.path.isfile(meta_file)):
        raise ValueError("Could not find meta file at: {}"
                         .format(meta_file))

    with open(meta_file) as json_file:
        meta_d = json.load(json_file)

    px_net_file = meta_d['px_model']
    px_params_file = meta_d['px_params']

    if not os.path.isabs(px_net_file):
      px_net_file = os.path.join(build_dir, px_net_file)

    if not os.path.isabs(px_params_file):
      px_params_file = os.path.join(build_dir, px_params_file)

    scheduled_xgraph = load(px_net_file, px_params_file)
    cb_scheduled_xgraph.copy_from(scheduled_xgraph)
