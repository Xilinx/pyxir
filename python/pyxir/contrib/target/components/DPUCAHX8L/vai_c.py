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

""" Module wrapping DPUCAHX8L VAI compiler """

import os
import json
import shutil
import warnings
import subprocess
import logging

from pyxir.shared.compiler_output import CompilerOutput
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.compiler.base_compiler import XGraphBaseCompiler
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner


logger = logging.getLogger('pyxir')

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class VAICompiler(XGraphBaseCompiler):

    """ Vitis-AI compiler wrapper for DPUCAHX8L """

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()

    def __init__(self,
                 xgraph,
                 arch,
                 work_dir=os.path.join(os.getcwd(), 'work'),
                 build_dir=os.getcwd(),
                 mode='debug'):

        super(VAICompiler, self).__init__(xgraph)

        if not os.path.isfile(arch):
            raise ValueError("Arch file: {} does not exist".format(arch))

        self.arch = arch


        q_output = self.xgraph.get_quantizer_output()
        self.netcfgs = {q_key: q_output.get_q_eval(q_key)
                        for q_key in q_output.keys()}
        assert(len(self.netcfgs) == 1)
        self.work_dir = work_dir

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.build_dir = build_dir if build_dir is not None else work_dir
        if not os.path.exists(self.build_dir):
            os.makedirs(self.build_dir)
        self.mode = mode
        self.c_output = CompilerOutput(name=xgraph.get_name())
        

    def compile(self) -> None:
        """ Start DPUCAHX8L compilation """

        net_name = list(self.netcfgs.keys())[0]
        netcfg = list(self.netcfgs.values())[0]

        # We only handle one partition at the moment
        Xp = VAICompiler.xgraph_partitioner\
            .get_subgraphs(self.xgraph)[0]
        subxg_layers = Xp.subgraph_data
        xgraph = VAICompiler.xgraph_factory.build_from_xlayer(subxg_layers)
        # assert xgraph.get_name() == net_name

        input_names = xgraph.get_input_names()
        input_shapes = [xgraph.get(in_name).shapes[:]
                        for in_name in input_names]
        output_names = list(Xp.attrs['__top_tensors'].keys()) # xgraph.get_output_names()
        output_shapes = [xgraph.get(out_name).shapes[:]
                         for out_name in output_names]

        if len(input_names) > 1:
            raise NotImplementedError("VAICompiler only handles models with"
                                      " one input at the moment but found: {}"
                                      .format(len(input_names)))

       
        command = """
        vai_c_tensorflow \
            --frozen_pb {} \
            --arch {} \
            --output_dir {} \
            --net_name {} \
            --options "{}"
        """.format(netcfg, self.arch, self.build_dir, net_name, str(dict()))

        logger.info("Command: {}".format(command))

        process = subprocess.Popen(command,
                                   shell=True,
                                   cwd=FILE_PATH,
                                   stdout=subprocess.PIPE)

        output, error = process.communicate()
        logger.debug("{} {}".format(output, error))


        if error is not None:
            error = error.decode('utf-8')
            raise ValueError(error)

        in_map = {in_name: in_name for in_name in input_names}
        out_map = {out_name: out_name for out_name in output_names}
        self.c_output.add(net_name, ['libvart-runner.so'], in_map, out_map)
        self.xgraph.set_compiler_output(self.c_output)

        # TODO
        self.xgraph.meta_attrs['compiled'] = True
        self.xgraph.meta_attrs['compiler_libs'] = ['libvart-runner.so']
        self.xgraph.meta_attrs['compiler_in_map'] = in_map
        self.xgraph.meta_attrs['compiler_out_map'] = out_map

        return self.xgraph
