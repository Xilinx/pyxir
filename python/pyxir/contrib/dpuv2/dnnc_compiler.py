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
Module wrapping DNNDK dnnc compiler


"""

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

from .dnnc_output import DNNCOutput

logger = logging.getLogger('pyxir')

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class DNNCCompiler(XGraphBaseCompiler):

    """
    TODO
    """
    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()

    def __init__(self,
                 xgraph,
                 dcf,
                 cpu_arch='arm64',
                 work_dir=os.path.join(os.getcwd(), 'work'),
                 mode='debug'):

        super(DNNCCompiler, self).__init__(xgraph)

        if not os.path.isfile(dcf):
            raise ValueError("Dcf file: {} does not exist".format(dcf))

        if cpu_arch != 'arm64':
            raise ValueError("Unsupported CPU architecture: {}. Supported"
                             " architectures are: 'arm64'")

        warnings.warn("This compilation only works with one network"
                      " configuration at the moment!!")

        q_output = self.xgraph.get_quantizer_output()
        self.netcfgs = {q_key: q_output.get_q_file(q_key)
                        for q_key in q_output.keys()}
        assert(len(self.netcfgs) == 1)
        self.dcf = dcf
        self.cpu_arch = cpu_arch
        self.work_dir = work_dir
        self.mode = mode
        self.c_output = CompilerOutput(name=xgraph.name)

    def compile(self):
        # type: () -> None
        """
        """

        net_name = list(self.netcfgs.keys())[0]
        netcfg = list(self.netcfgs.values())[0]

        xgraph = DNNCCompiler.xgraph_partitioner\
            .get_subgraphs(self.xgraph)[0].data
        assert(xgraph.name == net_name)

        input_names = xgraph.get_input_names()
        input_shapes = [xgraph.get(in_name).shapes[:]
                        for in_name in input_names]
        output_names = xgraph.get_output_names()
        output_shapes = [xgraph.get(out_name).shapes[:]
                         for out_name in output_names]

        if len(input_names) > 1:
            raise NotImplementedError("DNNCCompiler only handles models with"
                                      " one input at the moment but found: {}"
                                      .format(len(input_names)))
        # if len(output_names) > 1:
        #    raise NotImplementedError("DNNCCompiler only handles models with
        #       one output at the"\
        #        " moment but found: {}".format(len(output_names)))

        command = """
        dnnc --parser=tensorflow \
            --frozen_pb={} \
            --output_dir={} \
            --dcf={} \
            --cpu_arch={} \
            --mode={} \
            --net_name={} \
            --save_kernel \
            --dump all
        """.format(netcfg, self.work_dir, self.dcf, self.cpu_arch,
                   self.mode, net_name)

        logger.debug("Command: {}".format(command))

        process = subprocess.Popen(command.split(),
                                   cwd=FILE_PATH,
                                   stdout=subprocess.PIPE)
        output, error = process.communicate()

        if output is not None:
            output = output.decode('utf-8')
            
            logger.debug(output)

            do = DNNCOutput(str(repr(output)))

            dpu_input_nodes = do.get_input_nodes()
            dpu_output_nodes = do.get_output_nodes()

            in_shapes_log = ["{}*{}*{}".format(ishape[1], ishape[2], ishape[3])
                             for ishape in input_shapes]
            out_shapes_log = ["{}*{}*{}".format(os[1], os[2], os[3])
                              for os in output_shapes]

            in_map = {
                in_name: dpu_input_nodes[in_shape_str]
                for in_name, in_shape_str in
                zip(input_names, in_shapes_log)
            }
            out_map = {
                out_name: dpu_output_nodes[out_shape_str]
                for out_name, out_shape_str in
                zip(output_names, out_shapes_log)
            }

        if error is not None:
            error = error.decode('utf-8')
            raise ValueError(error)

        logger.debug("Output: {}".format(output))
        logger.debug("Error: {}".format(error))

        logger.debug("CROSS COMPILATION")
        command = """
        aarch64-linux-gnu-gcc -fPIC -shared {}/dpu_{}.elf -o {}/libdpumodel{}.so
        """.format(self.work_dir, net_name, self.work_dir, net_name)

        logger.debug("Command: {}".format(command))

        process = subprocess.Popen(command.split(),
                                   cwd=FILE_PATH,
                                   stdout=subprocess.PIPE)
        output, error = process.communicate()

        if output is not None:
            output = output.decode('utf-8')
        if error is not None:
            error = error.decode('utf-8')
            raise ValueError(error)

        logger.debug("Output: {}".format(output))
        logger.debug("Error: {}".format(error))

        lib_file = "{}/libdpumodel{}.so".format(self.work_dir, net_name)
        to_lib_file = "{}/libdpumodel{}.so".format(os.getcwd(), net_name)

        shutil.move(lib_file, to_lib_file)

        # { net_name: to_lib_file }
        self.c_output.add(net_name, [to_lib_file], in_map, out_map)

        self.xgraph.set_compiler_output(self.c_output)

        return self.xgraph
