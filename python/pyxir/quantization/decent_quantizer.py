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

"""Module wrapping the DECENT quantizer"""

import os
import json
import logging
import warnings
import subprocess
import numpy as np

from typing import List, Dict

from pyxir.graph import XGraph
from pyxir.contrib.tools import classification
from pyxir.shared.quantizer_output import QuantizerOutput
from pyxir.generator.tensorflow import TfGenerator
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.quantization.base_subgraph_quantizer import XGraphBaseSubgraphQuantizer
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner

FILE_PATH = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger("pyxir")


def is_decent_available():
    try:
        import tensorflow as tf

        return hasattr(tf, "contrib") and hasattr(tf.contrib, "decent_q")
    except ImportError:
        return False


class DECENTQuantizer(XGraphBaseSubgraphQuantizer):

    try:
        import tensorflow as tf

        if hasattr(tf, "contrib") and hasattr(tf.contrib, "decent_q"):
            from tensorflow.contrib import decent_q
        else:
            warnings.warn(
                "Could not import decent_q module. Please check" " if installed."
            )
    except ImportError:
        warnings.warn("Could not import decent_q module. Please check" " if installed.")

    xgraph_factory = XGraphFactory()
    xgraph_partitioner = XGraphPartitioner()

    def __init__(
        self,
        xgraph,
        inputs_func,
        work_dir=os.path.join(os.getcwd(), "work"),
        quant_iter=1,
        compiler_target=None,
        **kwargs
    ):

        super(DECENTQuantizer, self).__init__(xgraph, inputs_func, work_dir)

        self.quant_iter = quant_iter
        self.compiler_target = compiler_target
        self.gen = TfGenerator()
        self.partition_graphs = {}
        self.res = {}
        self.kwargs = kwargs

        self.q_output = QuantizerOutput(name=xgraph.get_name())

    def quantize_subgraph(
        self,
        xgraph: XGraph,
        inputs: Dict[str, np.ndarray],
        input_names: List[str],
        output_names: List[str],
    ) -> None:
        """Quantize subgraph with inputs"""

        # Import Tensorflow only when needed to avoid strict dependency
        import tensorflow as tf

        frozen_graph = self.partition_graphs[xgraph.get_name()]
        logger.info("Load frozen graph from: {}".format(frozen_graph))
        input_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(frozen_graph, "rb") as f:
            input_graph_def.ParseFromString(f.read())

        logger.info(
            "Quantization input: {} and output names: {}".format(
                input_names, output_names
            )
        )
        input_shapes = [X.shapes.tolist() for X in xgraph.get_input_layers()]

        in_batch_size = inputs[input_names[0]].shape[0]
        quant_batch_size = min(32, in_batch_size)
        nb_quant_iters = in_batch_size // quant_batch_size

        def inputs_func(iter):
            import numpy as np

            nonlocal inputs
            nonlocal quant_batch_size
            return {
                in_name: inputs[in_name][
                    iter * quant_batch_size : (iter + 1) * quant_batch_size
                ]
                for in_name in inputs.keys()
            }

        logger.info(
            "START decent quantization for graph partition: {}, nb_iters: {}, batch_size: {}".format(
                xgraph.get_name(), nb_quant_iters, quant_batch_size
            )
        )
        q_config = self.decent_q.QuantizeConfig(
            input_nodes=input_names,
            output_nodes=output_names,
            input_shapes=input_shapes,
            output_dir=self.work_dir,
            simulate_dpu="1",
            method="1",
            calib_iter=nb_quant_iters,
        )
        self.decent_q.quantize_frozen(input_graph_def, inputs_func, q_config)
        q_eval_file = os.path.join(self.work_dir, "quantize_eval_model.pb")

        self.q_output.add(
            xgraph.get_name(), frozen_graph, q_eval_file
        )



    def quantize(self) -> XGraph:
        """Quantize the XGraph model using the decent_q quantizer
        
        Returns
        -------
        q_xgraph: The quantized XGraph
        """

        # NOTE For Conv2Dtranspose layers we need the specific batch size in
        #   tensorflow 1.13
        batch_size = list(self.inputs_func(0).values())[0].shape[0]

        fs = self.gen.generate(
            self.xgraph,
            "graph",
            subgraphs_only=True,
            layout="NHWC",
            batch_size=batch_size,
            out_dir=self.work_dir,
            compiler_target=self.compiler_target,
            **self.kwargs
        )

        if len(fs) != 1:
            raise ValueError(
                "DECENT quantization currently only supports"
                " models with one DPU compatible partition,"
                " but got: {}".format(len(fs))
            )

        partition_key = list(fs.keys())[0]
        pb_path = list(fs.values())[0]

        self.partition_graphs[partition_key] = pb_path

        q_xgraph = super(DECENTQuantizer, self).quantize()

        self.xgraph.meta_attrs["is_quantized"] = True
        for qkey in self.q_output.keys():
            if "quant_keys" not in self.xgraph.meta_attrs:
                self.xgraph.meta_attrs["quant_keys"] = [qkey]
            else:
                self.xgraph.meta_attrs["quant_keys"].append(qkey)
            quant_orig_pb = self.q_output.get_orig_pb(qkey)
            quant_eval_file = self.q_output.get_q_eval(qkey)
            self.xgraph.meta_attrs[qkey] = {
                "orig_pb": quant_orig_pb,
                "q_eval": quant_eval_file,
            }

        return q_xgraph



    def eval(
        self,
        val_dir,
        gold_file,
        synset_words,
        batch_size,
        nb_batches,
        class_num=1000,
        gpu=0,
    ):
        #
        """
        """
        from progressbar import ProgressBar

        input_fn_data = {
            "prep_key": self.data_prep_key,
            "dir": val_dir,
            "batch": batch_size,
            "inputs": self.xgraph.get_input_names(),
        }

        with open(os.path.join(FILE_PATH, "calibration.json"), "w") as f:
            json.dump(input_fn_data, f)

        with open(gold_file) as f:
            val_set = [line.strip("\n").split(" ") for line in f.readlines()]

        # frozen_graph_file = os.path.join(os.getcwd(), 'test.pb')
        frozen_graph_file = os.path.join(self.output_dir, "quantize_eval_model.pb")
        # TODO
        assert len(self.xgraph.get_input_names()) == 1
        assert len(self.xgraph.get_output_names()) == 1
        input_node = self.xgraph.get_input_names()[0]
        output_node = self.xgraph.get_output_names()[0]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        input_graph_def = tf.Graph().as_graph_def()
        input_graph_def.ParseFromString(
            tf.gfile.FastGFile(frozen_graph_file, "rb").read()
        )

        tf.import_graph_def(input_graph_def, name="")

        # Get input tensors
        input_tensor = tf.get_default_graph().get_tensor_by_name(input_node + ":0")
        input_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, class_num])

        # Calculate accuracy
        output = tf.get_default_graph().get_tensor_by_name(output_node + ":0")
        prediction = tf.reshape(output, [batch_size, class_num])
        # correct_labels = tf.argmax(input_labels, 1)
        # top1_prediction = tf.nn.in_top_k(prediction, correct_labels, k = 1)
        # top5_prediction = tf.nn.in_top_k(prediction, correct_labels, k = 5)
        # top1_accuracy = tf.reduce_mean(tf.cast(top1_prediction,'float'))
        # top5_accuracy = tf.reduce_mean(tf.cast(top5_prediction,'float'))

        # Start evaluation
        logger.info("Start Evaluation for {} Batches...".format(nb_batches))
        with tf.Session() as sess:
            progress = ProgressBar()

            top1_sum_acc = 0
            top5_sum_acc = 0

            for iter in progress(range(0, nb_batches)):
                input_data = decent_prepfn.input_fn(iter)
                images = input_data[input_node]
                # labels = input_data['labels']
                logger.debug("IMAGES", images)
                labels = [
                    elem[1]
                    for elem in val_set[iter * batch_size : (iter + 1) * batch_size]
                ]
                feed_dict = {input_tensor: images}
                raw_predictions = sess.run(prediction, feed_dict)
                logger.debug(raw_predictions)

                # logger.debug("Predictions shape: {}"
                #              .format(raw_predictions.shape))
                # logger.debug("Labels length: {}".format(len(labels)))
                top_1 = classification.get_top_k_accuracy(
                    raw_predictions, synset_words, 1, labels
                )
                top_5 = classification.get_top_k_accuracy(
                    raw_predictions, synset_words, 5, labels
                )
                top1_sum_acc += top_1
                top5_sum_acc += top_5
                logger.debug("int: {}, {}".format(top_1, top_5))

        final_top1_acc = top1_sum_acc / nb_batches
        final_top5_acc = top5_sum_acc / nb_batches

        print("Accuracy: Top1: {}, Top5: {}".format(final_top1_acc, final_top5_acc))

    def dump(self, img_dir, input_names, max_dump_batches=1, dump_float=0):
        #
        """
        TODO: inupt_names
        """
        input_fn_data = {
            "prep_key": self.data_prep_key,
            "dir": img_dir,
            "batch": 1,
            "inputs": input_names,
        }

        with open(os.path.join(FILE_PATH, "calibration.json"), "w") as f:
            json.dump(input_fn_data, f)

        frozen_graph = os.path.join(self.output_dir, "quantize_eval_model.pb")

        command = """
        decent_q dump \
            --input_frozen_graph {} \
            --input_fn decent_prepfn.input_fn \
            --max_dump_batches {} \
            --dump_float {} \
            --output_dir {}
        """.format(
            frozen_graph, max_dump_batches, dump_float, self.output_dir
        )

        print("COMMAND", command)

        process = subprocess.Popen(
            command.split(), cwd=FILE_PATH, stdout=subprocess.PIPE
        )
        output, error = process.communicate()

        print(output, error)
