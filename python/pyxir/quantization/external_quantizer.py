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
Module wrapping DNNDK decent quantizer


"""

from abc import ABC
import os
import logging
import shutil

import tensorflow as tf
from tensorflow.contrib import decent_q
from tensorflow.contrib.decent_q.python.utils import *
from tensorflow.contrib.decent_q.python.quantize_graph import *


from pyxir.shared.quantizer_output import QuantizerOutput
from pyxir.generator.tensorflow import TfGenerator
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.quantization.base_subgraph_quantizer import XGraphBaseSubgraphQuantizer
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner


FILE_PATH = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger('pyxir')


class ExternalQuantizer(XGraphBaseSubgraphQuantizer, ABC):

    xgraph_factory = XGraphFactory()
    xgraph_partitioner = XGraphPartitioner()

    def __init__(self,
                 xgraph,
                 inputs_func,
                 work_dir=os.path.join(os.getcwd(), 'work')):

        super(ExternalQuantizer, self).__init__(xgraph, inputs_func, work_dir)

        self.gen = TfGenerator()
        self.partition_graphs = {}
        self.res = {}
        self.q_output = QuantizerOutput(name=xgraph.get_name())

    def _propagate_quant_info(self, xgraph):
        # setup empty vqi and vqo for every layer w/o vai_quant
        for layer in xgraph.get_layers():
            if 'vai_quant' not in layer.attrs:
                layer.attrs['vai_quant'] = ['vai_quant_in', 'vai_quant_out']
                layer.attrs['vai_quant_in'] = ''
                layer.attrs['vai_quant_out'] = ''
        # for every layer
        for layer in xgraph.get_layers():
            # if the layer has non empty vqo, propagate it to the output layers
            if layer.attrs['vai_quant_out'] != '':
                l_vqo = layer.attrs['vai_quant_out']
                # for every output layer
                for t_idx, t_name in enumerate(layer.tops):
                    t_layer = xgraph.get(t_name)
                    # if the input quant is not specified in the output layer
                    if t_layer.attrs['vai_quant_in'] == '':
                        # get quant info from current layer, two by two
                        t_vqi = [l_vqo[2 * t_idx], l_vqo[2 * t_idx + 1]]
                        t_layer.attrs['vai_quant_in'] = t_vqi
            # if the layer has non empty vqi, propagate it to the input layers
            if layer.attrs['vai_quant_in'] != '':
                l_vqi = layer.attrs['vai_quant_in']
                # for every input layer
                for b_idx, b_name in enumerate(layer.bottoms):
                    b_layer = xgraph.get(b_name)
                    if b_layer.attrs['vai_quant_out'] == '':
                        b_vqo = [l_vqi[2 * b_idx], l_vqi[2 * b_idx + 1]]
                        b_layer.attrs['vai_quant_out'] = b_vqo

    def quantize(self):
        # NOTE For Conv2Dtranspose layers we need the specific batch size in tensorflow 1.13
        batch_size = list(self.inputs_func(0).values())[0].shape[0]
        fs = self.gen.generate(
            self.xgraph,
            'graph',
            subgraphs_only=True,
            layout='NHWC',
            batch_size=batch_size)
        assert len(fs) == 1, 'Too many partitions'
        partition_key = list(fs.keys())[0]
        pb_path = list(fs.values())[0]
        self.partition_graphs[partition_key] = pb_path

        q_xgraph = super(ExternalQuantizer, self).quantize()

        self.xgraph.meta_attrs["is_quantized"] = True
        for qkey in self.q_output.keys():
            if 'quant_keys' not in self.xgraph.meta_attrs:
                self.xgraph.meta_attrs['quant_keys'] = [qkey]
            else:
                self.xgraph.meta_attrs['quant_keys'].append(qkey)
            quant_file = self.q_output.get_q_file(qkey)
            quant_info_file = self.q_output.get_q_info(qkey)
            quant_orig_pb = self.q_output.get_orig_pb(qkey)
            self.xgraph.meta_attrs[qkey] = {
                'q_file': quant_file,
                'q_info': quant_info_file,
                'orig_pb': quant_orig_pb}
        return q_xgraph


class ExternalQuantizerTxtOutput(ExternalQuantizer):

    def _get_quant_info(self, xgraph):
        lines = []
        # extract annotations from op layers
        for idx, layer in enumerate(xgraph.get_layers()):
            # layers are 1-indexed in the str format
            line = [str(idx + 1), layer.name]
            assert "vai_quant" in layer.attrs, 'Every layer must have quant annotation'
            for quant_elem in layer.attrs['vai_quant']:
                line.extend([str(i) for i in layer.attrs[quant_elem]])
            lines.append(line)
        return lines

    def _save_quant_info(self, quant_info, filename):
        s = '\n'.join([' '.join(line) for line in quant_info])
        with open(filename, 'w') as f:
            f.write(s)

    def gen_output(self, xgraph):
        quant_info = self._get_quant_info(xgraph)
        quant_info_file = os.path.join(self.work_dir, f'quant_info_{xgraph.get_name()}.txt')
        self._save_quant_info(quant_info, quant_info_file)
        return quant_info_file

    def quantize_subgraph(self, xgraph, inputs, input_names, output_names):
        # Load tensorflow subgraph converted from xgraph
        frozen_graph = self.partition_graphs[xgraph.get_name()]
        input_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            input_graph_def.ParseFromString(f.read())
        # Propagate quantization annotations within the xgraph
        self._propagate_quant_info(xgraph)
        # gen quantization output
        quant_output = self.gen_output(xgraph)
        # Set output of quantizer
        self.q_output.add(xgraph.get_name(), 'NULL', quant_output, frozen_graph)


class ExternalQuantizerDecentOutput(ExternalQuantizer):

    def _gen_quant_temp_data(self, xgraph):
        temp_path = os.path.join(self.work_dir, "temp")
        os.makedirs(temp_path) if not os.path.exists(temp_path) else True
        files = {}
        for layer in xgraph.get_layers():
            assert "vai_quant" in layer.attrs, 'Every layer must have quant annotation'
            vai_quant_str_list = [str(a) for a in layer.attrs['vai_quant']]
            for quant_elem in vai_quant_str_list:
                if quant_elem == 'vai_quant_weights':
                    file_name = f"{layer.name}_filter_wquant"
                    content = [f"{layer.name}/filter/wquant"]
                    content.extend([str(i) for i in layer.attrs[quant_elem]])
                    files[file_name] = content
                elif quant_elem == 'vai_quant_biases':
                    file_name = f"{layer.name}_Bias_bias_wquant"
                    content = [f"{layer.name}_Bias/bias/wquant"]
                    content.extend([str(i) for i in layer.attrs[quant_elem]])
                    files[file_name] = content
                elif quant_elem == 'vai_quant_out':
                    if 'vai_quant_biases' in vai_quant_str_list:
                        file_name = f"{layer.name}_Bias_aquant"
                        content = [f"{layer.name}_Bias/aquant"]
                        content.extend([str(i) for i in layer.attrs[quant_elem]])
                        files[file_name] = content
                    else:
                        file_name = f"{layer.name}_aquant"
                        content = [f"{layer.name}/aquant"]
                        content.extend([str(i) for i in layer.attrs[quant_elem]])
                        files[file_name] = content
        for file_name, file_content in files.items():
            with open(os.path.join(self.work_dir, 'temp', file_name), 'w') as f:
                f.write(' '.join(file_content))

    # Some nodes don't get the correct value from temp, so we pass through all of them again
    def set_deploy_graph_quant(self, xgraph, deploy_graph_def) -> None:
        for idx, dnode in enumerate(deploy_graph_def.node):
            assert dnode.name in xgraph, f'Deploy node {dnode.name} not found'
            xnode = xgraph.get(dnode.name)
            if 'ipos' in dnode.attr.keys():
                quant_in = xnode.attrs['vai_quant_in']
                dnode.attr['ipos'].list.i[:] = quant_in
            if 'opos' in dnode.attr.keys():
                quant_out = xnode.attrs['vai_quant_out']
                dnode.attr['opos'].list.i[:] = quant_out
            if 'bpos' in dnode.attr.keys():
                quant_biases = xnode.attrs['vai_quant_biases']
                dnode.attr['bpos'].list.i[:] = quant_biases
            if 'wpos' in dnode.attr.keys():
                quant_weights = xnode.attrs['vai_quant_weights']
                dnode.attr['wpos'].list.i[:] = quant_weights

    def quantize_frozen(self, xgraph, input_graph_def, q_config):
        calib_graph_def = CreateQuantizeCalibrationGraphDef(input_graph_def, q_config)
        quantize_eval_graph_def = CreateQuantizeEvaluationGraphDef(calib_graph_def, q_config)
        q_config.output_nodes = get_quantized_nodes(quantize_eval_graph_def, q_config.output_nodes)
        deploy_graph_def = CreateQuantizeDeployGraphDef(quantize_eval_graph_def, q_config)
        self.set_deploy_graph_quant(xgraph, deploy_graph_def)
        quantize_deploy_graph_path = os.path.join(q_config.output_dir, "deploy_model.pb")
        save_pb_file(deploy_graph_def, quantize_deploy_graph_path)
        return quantize_deploy_graph_path

    def quantize_subgraph(self, xgraph, inputs, input_names, output_names):
        # Load tensorflow subgraph converted from xgraph
        frozen_graph = self.partition_graphs[xgraph.get_name()]
        input_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            input_graph_def.ParseFromString(f.read())
        # Propagate quantization annotations within the xgraph
        self._propagate_quant_info(xgraph)
        # Generate decent_q output
        input_shapes = [l.shapes.tolist() for l in xgraph.get_input_layers()]
        q_config = decent_q.QuantizeConfig(
            input_nodes=input_names,
            output_nodes=output_names,
            input_shapes=input_shapes,
            output_dir=self.work_dir,
            method='1',
            calib_iter=0)
        self._gen_quant_temp_data(xgraph)
        quant_deploy_path = self.quantize_frozen(xgraph, input_graph_def, q_config)
        self.q_output.add(xgraph.get_name(), quant_deploy_path, '', frozen_graph)



