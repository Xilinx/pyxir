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

""" Module for generating a tensorflow graph from an XGraph """

import os
import logging

# import tensorflow as tf

from pyxir.graph.optimization import optimizations, conditions
from pyxir.graph.optimization.xgraph_optimization_pass \
    import XGraphOptimizationPass
from pyxir.graph.optimization.optimizers.basic_optimizer \
    import XGraphBasicOptimizer

from pyxir.runtime.runtime_factory import RuntimeFactory
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.transformers.layout_transformation_pass \
    import XGraphLayoutTransformationPass

logger = logging.getLogger('pyxir')


class XGraphTfGeneratorOptimizer(XGraphBasicOptimizer):

    def __init__(self, xgraph, copy=False):
        super(XGraphTfGeneratorOptimizer, self).__init__(xgraph)

        # CONV/BIAS/BN/SCALE merge optimization
        opt_pass = XGraphOptimizationPass(
            name='XDNN-OptimizationPass-2-Merge_Conv_Bias_BN_Scale',
            output_png='after_merge_conv_bias_bn_scale.png',
            repeat_until_stable=True
        )

        logger.info("Add MergeBiasIntoConvDense pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs:
                'Eltwise' in X.type and X.data is not None,
            opt_func=optimizations.merge_bias,
            name='MergeBiasIntoConvDense'
        )

        logger.info("Add MergeBNIntoConv pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'BatchNorm' in X.type,
            opt_func=optimizations.merge_batchnorm_into_conv,
            name='MergeBNIntoConv'
        )

        logger.info("Add MergeScaleIntoConvBN pass")
        opt_pass.add_optimization(
            condition_func=lambda bXs, X, tXs: 'Scale' in X.type,
            opt_func=optimizations.merge_scale_into_conv_bn,
            name='MergeScaleIntoConvBN'
        )
        self.add_optimization_pass(20, opt_pass)


class TfGenerator(object):

    """
    Responsible for generating tensorflow model from xgraph data structure
    """

    runtime_factory = RuntimeFactory()
    xgraph_factory = XGraphFactory()
    xgraph_partitioner = XGraphPartitioner()

    @classmethod
    def generate(cls, xgraph, base_name, subgraphs_only=False, layout='NCHW',
                 batch_size=-1, placeholder=False, out_dir=os.getcwd(), **kwargs):
        # type: (XGraph, str, bool, str, int, bool, str, dict) -> Dict[str, str]
        """
        Generate one or multiple tensorflow pb file from an xgraph and
        return dictionary of the base_name/partitions mapping to the pb files
        """
        # Import tensorflow only when needed
        import tensorflow as tf

        executors = []
        if not subgraphs_only:
            executors.append(
                (base_name, base_name,
                 TfGenerator.runtime_factory.build_runtime(
                     xgraph, batch_size=batch_size, placeholder=placeholder))
            )
        else:
            for Xp in \
                    TfGenerator.xgraph_partitioner.get_subgraphs(xgraph):

                out_tensor_names = list(Xp.attrs['__top_tensors'].keys())
                sub_xgraph = TfGenerator.xgraph_factory.build_from_xlayer(
                    Xp.subgraph_data)
                executors.append(
                    (base_name + '_' + Xp.name,
                     Xp.name,
                     TfGenerator.runtime_factory
                        .build_runtime(sub_xgraph,
                                       batch_size=batch_size,
                                       placeholder=placeholder,
                                       out_tensor_names=out_tensor_names,
                                       **kwargs),
                     out_tensor_names), # sub_xgraph.get_output_names()
                )

        ret = {}
        for file_name, name, executor, output_names in executors:
            graph_def = executor.tf_graph.as_graph_def()
            # with executor.tf_step_graph.as_default():
            #    graph_def = tf.get_default_graph().as_graph_def()

            with tf.compat.v1.Session(graph=executor.tf_graph) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                graph_def = tf.graph_util.convert_variables_to_constants(
                    sess,
                    graph_def,
                    output_names
                )

            file_path = os.path.join(out_dir, file_name + '.pb')

            with tf.gfile.GFile(file_path, "wb") as f:
                f.write(graph_def.SerializeToString())

            ret[name] = file_path

        return ret

    @classmethod
    def run(self, xgraph, pb_file, inputs):
        # type: (XGraph, str, dict[str, numpy.ndarray])
        """ Run frozen tensorflow graph for corresponding XGraph """
       
        # Import Tensorflow only when needed
        import tensorflow as tf

        input_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            input_graph_def.ParseFromString(f.read())

        tf.compat.v1.reset_default_graph()
        tf.import_graph_def(input_graph_def, name='')

        input_names = xgraph.get_input_names()
        output_names = xgraph.get_output_names()

        inputs_tf = {}
        for in_name in input_names:
            input_tensor = tf.get_default_graph().get_tensor_by_name(
                                                    in_name + ':0')

            inputs_tf[input_tensor] = inputs[in_name]

        outputs = [tf.get_default_graph().get_tensor_by_name(out_name + ':0')
                   for out_name in output_names]

        with tf.compat.v1.Session() as sess:
            res = sess.run(outputs, feed_dict=inputs_tf)

        return res
