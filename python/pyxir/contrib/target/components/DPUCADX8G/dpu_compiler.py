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

"""Module wrapping DNNDK dnnc compiler"""

import os
import json
import shutil
import warnings
import subprocess

import logging

from pyxir.shared.compiler_output import CompilerOutput
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.generator.tensorflow import TfGenerator
from pyxir.compiler.base_compiler import XGraphBaseCompiler
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.transformers.layout_transformation_pass import \
    XGraphLayoutTransformationPass

logger = logging.getLogger('pyxir')

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class DPUCompiler(XGraphBaseCompiler):
    """Wrapper around DPUCADX8G compiler"""
    
    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()
    tf_generator = TfGenerator()

    def __init__(self,
                 xgraph,
                 target,
                 arch,
                 work_dir=os.path.join(os.getcwd(), 'work'),
                 build_dir=None,
                 mode='debug'):

        super(DPUCompiler, self).__init__(xgraph)

        if not os.path.isfile(arch):
            raise ValueError("Arch file: {} does not exist".format(arch))

        q_output = self.xgraph.get_quantizer_output()
        self.netcfgs = {q_key: q_output.get_orig_pb(q_key)
                        for q_key in q_output.keys()}
        self.quant_info = {q_key: q_output.get_q_info(q_key)
                           for q_key in q_output.keys()}
        assert(len(self.netcfgs) == 1)
        self.work_dir = work_dir
        self.build_dir = build_dir if build_dir is not None else work_dir
        self.target = target
        self.arch = arch
        self.mode = mode
        self.c_output = CompilerOutput(name=xgraph.get_name())

    def Getopts(self, input_shapes):
        return {
            "maximumasrelu": True,
            "pipelineconvmaxpool":False,
            "bytesperpixels": 1,
            "dsp": 96,
            "memory": 9,
            "ddr": "256",
            "cpulayermustgo": True,
            "forceweightsfullyconnected": True,
            "mixmemorystrategy": True,
            "maximumasrelu": True,
            "pipelineconvmaxpool": True,
            'bridges': ['bytype', 'Concat'],
            "usedeephi": True,
            'placeholdershape': input_shapes
        }

    def compile(self):
        # type: () -> None
        """ """
        layout_transform_pass = \
            XGraphLayoutTransformationPass('NHWC', target=self.target)
        self.xgraph = layout_transform_pass.execute(self.xgraph,
                                                    subgraphs_only=False)
        
        # netcfg = list(self.netcfgs.values())[0]  # orig pb file
        quant_info_file = list(self.quant_info.values())[0]  # quant info file

        Xp = DPUCompiler.xgraph_partitioner\
            .get_subgraphs(self.xgraph)[0]
        subxg_layers = Xp.subgraph_data
        xgraph = DPUCompiler.xgraph_factory.build_from_xlayer(subxg_layers)
        net_name = list(self.netcfgs.keys())[0]
        fs = DPUCompiler.tf_generator.generate(self.xgraph,
                                               'graph',
                                               subgraphs_only=True,
                                               layout='NHWC',
                                               batch_size=1,
                                               placeholder=True,
                                               out_dir=self.work_dir,
                                               # kwargs
                                               compiler_target='DPUv1Compiler')
        netcfg = list(fs.values())[0]

        input_names = xgraph.get_input_names()
        input_shapes = [xgraph.get(in_name).shapes.tolist()[:]
                        for in_name in input_names]
        output_names = list(Xp.attrs['__top_tensors'].keys()) # xgraph.get_output_names()
        output_shapes = [xgraph.get(out_name).shapes.tolist()[:]
                         for out_name in output_names]
        if len(input_names) > 1:
            raise NotImplementedError("DPUCompiler only handles models with"
                                      " one input at the moment but found: {}"
                                      .format(len(input_names)))
        opt_input_shapes = {in_name: [e if e != -1 else 1 for e in input_shape]
                            for in_name, input_shape
                            in zip(input_names, input_shapes)}
        opts = self.Getopts(opt_input_shapes)
        if not os.path.isfile(quant_info_file):
            raise ValueError("quant file: {} does not exist"
                             .format(quant_info_file))
        opts['quant_cfgfile'] = quant_info_file
        opts = str(opts)
        command = """
            vai_c_tensorflow \
            --frozen_pb {} \
            --arch {} \
            --output_dir {} \
            --net_name {}\
            --options "{}"
        """.format(netcfg, self.arch, self.build_dir, 'compiler', opts)
        logger.info("command: {}".format(command))
        process = subprocess.Popen(command,
                                   shell=True,
                                   cwd=FILE_PATH,
                                   stdout=subprocess.PIPE)
        output, error = process.communicate()
        if output is not None:
            output = output.decode('utf-8')
            if 'SUCCESSFUL COMPILATION' not in output:
                logger.info(output)
                raise ValueError('compiler is failed. Please see the log for'
                                 ' more details')
        if error is not None:
            error = error.decode('utf-8')
            # raise ValueError(error)

        logger.debug("Output: {}".format(output))
        logger.debug("Error: {}".format(error))
        compiler_json_file = self.build_dir + '/compiler.json'
        with open(compiler_json_file) as json_file:
            json_graph = json.load(json_file)
        graph_inputs = json_graph["inputs"]
        graph_outputs = json_graph["outputs"]
        logger.debug("{} {}".format(input_names, graph_inputs))
        logger.debug("{} {}".format(output_names, graph_outputs))

        in_map = {in_name: in_name for in_name in input_names}
        compiler_map = {layer['name']: layer for layer in json_graph['network']}
        out_nodes = [graph_output['previous_layers'][0] for graph_output in graph_outputs]
        out_nodes_merged = [compiler_map[out_name]['merged'][-1] for out_name in out_nodes]
        in_map = {in_name: in_name for in_name in input_names}
        out_map = {out_name: out_nodes[out_nodes_merged.index(out_name)] for out_name in output_names}

        self.c_output.add(net_name, ['dpuv1lib.so'], in_map, out_map)
        self.xgraph.set_compiler_output(self.c_output)

        # TODO
        self.xgraph.meta_attrs['compiled'] = True
        self.xgraph.meta_attrs['compiler_libs'] = ['dpuv1lib.so']
        self.xgraph.meta_attrs['compiler_in_map'] = in_map
        self.xgraph.meta_attrs['compiler_out_map'] = out_map

        return self.xgraph
