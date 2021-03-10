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

""" Module wrapping DPUCZDX8G VAI compiler """

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


class VAICompilerDNNC(XGraphBaseCompiler):

    """ Vitis-AI compiler wrapper for DPUCZDX8G DNNC compiler"""

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()

    def __init__(self,
                 xgraph,
                 arch,
                 meta,
                 dcf,
                 cpu_arch='arm64',
                 work_dir=os.path.join(os.getcwd(), 'work'),
                 build_dir=os.getcwd(),
                 mode='debug'):

        super(VAICompilerDNNC, self).__init__(xgraph)

        if not os.path.isfile(arch):
            raise ValueError("Arch file: {} does not exist".format(arch))

        if cpu_arch != 'arm64':
            raise ValueError("Unsupported CPU architecture: {}. Supported"
                             " architectures are: 'arm64'")

        q_output = self.xgraph.get_quantizer_output()
        self.netcfgs = {q_key: q_output.get_q_file(q_key)
                        for q_key in q_output.keys()}
        assert(len(self.netcfgs) == 1)
        self.arch = arch
        self.meta = meta
        self.dcf = dcf
        self.cpu_arch = cpu_arch
        self.work_dir = work_dir

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.build_dir = build_dir if build_dir is not None else work_dir
        if not os.path.exists(self.build_dir):
            os.makedirs(self.build_dir)
        self.mode = mode
        self.c_output = CompilerOutput(name=xgraph.get_name())
        

    def compile(self) -> None:
        """ Start DPUv2 compilation """

        net_name = list(self.netcfgs.keys())[0]
        netcfg = list(self.netcfgs.values())[0]

        # We only handle one partition at the moment
        Xp = VAICompilerDNNC.xgraph_partitioner\
            .get_subgraphs(self.xgraph)[0]
        subxg_layers = Xp.subgraph_data
        xgraph = VAICompilerDNNC.xgraph_factory.build_from_xlayer(subxg_layers)
        # assert xgraph.get_name() == net_name

        input_names = xgraph.get_input_names()
        input_shapes = [xgraph.get(in_name).shapes[:]
                        for in_name in input_names]
        output_names = list(Xp.attrs['__top_tensors'].keys()) # xgraph.get_output_names()
        output_shapes = [xgraph.get(out_name).shapes[:]
                         for out_name in output_names]

        if len(input_names) > 1:
            raise NotImplementedError("VAICompilerDNNC only handles models with"
                                      " one input at the moment but found: {}"
                                      .format(len(input_names)))

        #command = """
        #vai_c_tensorflow \
        #    --frozen_pb {} \
        #    --arch {} \
        #    --output_dir {} \
        #    --net_name {} \
        #    --options "{}"
        #""".format(netcfg, self.arch, self.work_dir, net_name, str(dict()))
        # import pdb; pdb.set_trace()

        command = """
        dnnc-dpuv2 --parser tensorflow\
            --frozen_pb {} \
            --cpu_arch {} \
            --output_dir {} \
            --net_name {} \
            --dcf {}
        """.format(netcfg, self.cpu_arch, self.work_dir, net_name, self.dcf)


        logger.info("Command: {}".format(command))

        process = subprocess.Popen(command,
                                   shell=True,
                                   cwd=FILE_PATH,
                                   stdout=subprocess.PIPE)

        output, error = process.communicate()
        logger.debug("{} {}".format(output, error))

        if output is not None:
            output = output.decode('utf-8')

            logger.info("Output: {}".format(output))
            logger.info("Output names: {}".format(output_names))

            do = DNNCOutput(str(repr(output)))

            dpu_input_nodes = do.get_input_nodes()
            dpu_output_nodes = do.get_output_nodes()
            dpu_output_nodes_on_shapes = do.get_output_nodes_on_shapes()

            in_shapes_log = ["{}*{}*{}".format(ishape[1], ishape[2], ishape[3])
                             for ishape in input_shapes]
            out_shapes_log = ["{}*{}*{}".format(os[1], os[2], os[3])
                              for os in output_shapes]

            in_map = {in_name: in_name + ':0' for in_name, _ in zip(input_names, in_shapes_log)}
            out_map = {}

            for out_name, out_shape_str in zip(output_names, out_shapes_log):
                # DNNC changes naming
                dnnc_out_name = do.get_dnnc_str(out_name)
                if dnnc_out_name in dpu_output_nodes:
                    out_map[out_name] = dpu_output_nodes[dnnc_out_name]
                # out_name: dpu_output_nodes[out_shape_str] + ':0'
                else:
                    assert len(dpu_output_nodes_on_shapes) == len(output_names),\
                        "Can't retrieve right out tensor names from DNNC compiler output"
                    out_map[out_name] = dpu_output_nodes_on_shapes[out_shape_str]

            logger.info("DPU kernel in_map: {}".format(in_map))
            logger.info("DPU kernel out_map: {}".format(out_map))

        if error is not None:
            error = error.decode('utf-8')
            raise ValueError(error)

        logger.info("VAI_C Output: {}".format(output))
        logger.info("VAI_C Error: {}".format(error))

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
        to_lib_file = "{}/libdpumodel{}.so".format(self.build_dir, net_name)
        shutil.move(lib_file, to_lib_file)

        # meta_file = "{}/meta.json".format(self.work_dir)
        self.meta["vitis_dpu_kernel"] = net_name
        to_meta_file = "{}/meta.json".format(self.build_dir)
        # shutil.move(meta_file, to_meta_file)

        with open(to_meta_file, 'w') as f:
            json.dump(self.meta, f)

        self.c_output.add(net_name, [to_lib_file], in_map, out_map)

        self.xgraph.set_compiler_output(self.c_output)

        return self.xgraph
