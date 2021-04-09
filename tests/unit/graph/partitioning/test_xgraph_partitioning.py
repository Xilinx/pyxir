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

"""Module for testing the xgraph partitioning functionality"""

import os
import unittest

import numpy as np

# ! Important for device registration
import pyxir as px

from pyxir.graph.layer.xlayer import XLayer, ConvData, BatchData
from pyxir.graph.partitioning.xgraph_partitioner import XGraphPartitioner
from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.target_registry import TargetRegistry, register_op_support_check

import logging

logging.basicConfig()
logger = logging.getLogger("pyxir")
# logger.setLevel(logging.DEBUG)


class TestXGraphPartitioner(unittest.TestCase):

    xgraph_partitioner = XGraphPartitioner()
    xgraph_factory = XGraphFactory()

    @classmethod
    def setUpClass(cls):
        def xgraph_build_func(xgraph):
            raise NotImplementedError("")

        def xgraph_optimizer(xgraph):
            raise NotImplementedError("")

        def xgraph_quantizer(xgraph):
            raise NotImplementedError("")

        def xgraph_compiler(xgraph):
            raise NotImplementedError("")

        target_registry = TargetRegistry()
        target_registry.register_target(
            "test",
            xgraph_optimizer,
            xgraph_quantizer,
            xgraph_compiler,
            xgraph_build_func,
        )

        @register_op_support_check("test", "Convolution")
        def conv_op_support(X, bXs, tXs):
            return True

        @register_op_support_check("test", "Pooling")
        def pooling_op_support(X, bXs, tXs):
            return True

        @register_op_support_check("test", "Concat")
        def concat_op_support(X, bXs, tXs):
            return False

        @register_op_support_check("test", "Eltwise")
        def eltwise_op_support(X, bXs, tXs):
            return True

        @register_op_support_check("test", "ReLU")
        def relu_op_support(X, bXs, tXs):
            return True

    @classmethod
    def tearDownClass(cls):

        target_registry = TargetRegistry()
        target_registry.unregister_target("test")

    def test_basic(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        x2 = px.ops.input("in2", shape=[1, 2, 2, 2])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv = px.ops.conv2d(
            op_name="conv1",
            input_layer=x1,
            weights_layer=w1,
            kernel_size=[2, 2],
            strides=[1, 1],
            padding_hw=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=2,
            data_layout="NCHW",
            kernel_layout="OIHW",
        )
        pool = px.ops.pool2d(
            op_name="pool1", input_layer=conv, pool_type="Avg", pool_size=[2, 2],
        )
        add = px.ops.eltwise("add1", pool, x2)
        net = [x1, x2, conv, pool, add]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 5
        assert p_xgraph.get_subgraph_names() == ["xp0"]

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Pooling"]
        assert p_xlayers[3].type[0] in ["Input"]
        assert p_xlayers[4].type[0] in ["Eltwise"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"
        assert p_xlayers[3].target == "cpu"
        assert p_xlayers[4].target == "test"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp0"
        assert p_xlayers[2].subgraph == "xp0"
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph == "xp0"

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1", "in2"]
        assert xp0.tops == []
        assert xp0.shapes == [[-1, 2, 2, 2]]
        assert xp0.sizes == [8]

        assert len(xp0_xgraph) == 5
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == "Input"
        assert xp0_layers[0].layer[0] == "conv1"
        assert xp0_layers[1].type[0] == "Convolution"
        assert xp0_layers[2].type[0] == "Pooling"
        assert xp0_layers[3].type[0] == "Input"
        assert xp0_layers[4].type[0] == "Eltwise"

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ["conv1"]
        assert xp0_layers[1].bottoms == ["xinput0"]
        assert xp0_layers[1].tops == ["pool1"]
        assert xp0_layers[2].bottoms == ["conv1"]
        assert xp0_layers[2].tops == ["add1"]

    def test_interrupt_partition_in_add_branch(self):
        x = px.ops.input("in1", shape=[1, 28, 28, 2028])
        w1 = px.ops.constant("weight", np.ones((2048, 2048, 1, 1), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1",
            input_layer=x,
            weights_layer=w1,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding_hw=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=2048,
            data_layout="NHWC",
            kernel_layout="OIHW",
        )
        r1 = px.ops.relu("r1", [conv1])
        w2 = px.ops.constant("weight", np.ones((512, 2048, 1, 1), dtype=np.float32))
        conv2 = px.ops.conv2d(
            op_name="conv2",
            input_layer=r1,
            weights_layer=w2,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding_hw=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=512,
            data_layout="NHWC",
            kernel_layout="OIHW",
        )
        sigm = px.ops.sigmoid("sigm", [conv2])  # Unsupported layer
        w3 = px.ops.constant("weight", np.ones((2048, 512, 1, 1), dtype=np.float32))
        conv3 = px.ops.conv2d(
            op_name="conv3",
            input_layer=sigm,
            weights_layer=w3,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding_hw=[0, 0, 0, 0],
            dilation=[1, 1],
            groups=1,
            channels=2048,
            data_layout="NHWC",
            kernel_layout="OIHW",
        )  # Although this layer is supported, it should not be in the partition
        add = px.ops.eltwise(
            "add", r1, conv3
        )  # Although this layer is supported, it should not be in the partition
        net = [x, conv1, r1, conv2, sigm, conv3, add]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 7
        assert p_xgraph.get_subgraph_names() == ["xp0"]

        p_xlayers = p_xgraph.get_layers()

        assert p_xgraph.get("in1").target == "cpu"
        assert p_xgraph.get("conv1").target == "test"
        assert p_xgraph.get("r1").target == "test"
        assert p_xgraph.get("conv2").target == "test"
        assert p_xgraph.get("sigm").target == "cpu"
        assert p_xgraph.get("conv3").target == "cpu"
        assert p_xgraph.get("add").target == "cpu"

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1"]
        assert xp0.tops == ["add", "sigm"]
        assert xp0.shapes == [[-1, 28, 28, 2048], [-1, 28, 28, 512]]
        assert xp0.sizes == [28 * 28 * 2048, 28 * 28 * 512]

        assert len(xp0_xgraph) == 4
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == "Input"
        assert xp0_layers[0].layer[0] == "conv1"
        assert xp0_layers[1].type[0] == "Convolution"
        assert xp0_layers[2].type[0] == "ReLU"
        assert xp0_layers[3].type[0] == "Convolution"

    def test_complete_partition(self):
        x = px.ops.input("in1", shape=[1, 1, 4, 4])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv = px.ops.conv2d(
            op_name="conv1", input_layer=x, weights_layer=w1, kernel_size=[2, 2],
        )
        pool = px.ops.pool2d(
            op_name="pool1", input_layer=conv, pool_type="Avg", pool_size=[2, 2],
        )
        net = [x, conv, pool]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 3
        assert p_xgraph.get_subgraph_names() == ["xp0"]

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Pooling"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp0"
        assert p_xlayers[2].subgraph == "xp0"

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1"]
        assert xp0.tops == []
        assert xp0.shapes == [[-1, 2, 2, 2]]
        assert xp0.sizes == [8]
        assert xp0.attrs["target"] == "test"
        assert xp0.attrs["__bottom_tensors"] == {"xinput0": ["in1"]}
        assert xp0.attrs["orig_bottom_tensors"] == {"xinput0": ["in1"]}
        assert xp0.attrs["__top_tensors"] == {"pool1": []}
        assert xp0.attrs["orig_top_tensors"] == {"pool1": []}

        assert len(xp0_xgraph) == 3
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == "Input"
        assert xp0_layers[0].layer[0] == "conv1"
        assert xp0_layers[1].type[0] == "Convolution"
        assert xp0_layers[2].type[0] == "Pooling"

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ["conv1"]
        assert xp0_layers[1].bottoms == ["xinput0"]
        assert xp0_layers[1].tops == ["pool1"]
        assert xp0_layers[2].bottoms == ["conv1"]
        assert xp0_layers[2].tops == []

    def test_two_partitions_through_interruption(self):
        # A layer inside a residual type branch os not supported
        # Here: BatchNorm
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=x1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 2, 3, 3
        pool = px.ops.pool2d(
            op_name="pool1",
            input_layer=conv1,
            pool_type="Avg",
            pool_size=[2, 2],
            padding=[1, 1, 0, 0],
        )  # 1, 2, 3, 3
        bn_mean = px.ops.constant("mean", np.ones((2,), dtype=np.float32))
        bn_var = px.ops.constant("var", np.ones((2,), dtype=np.float32))
        bn_gamma = px.ops.constant("gamma", np.ones((2,), dtype=np.float32))
        bn_beta = px.ops.constant("beta", np.ones((2,), dtype=np.float32))
        bn = px.ops.batch_norm(
            op_name="bn1",
            input_layer=conv1,
            mean_layer=bn_mean,
            variance_layer=bn_var,
            gamma_layer=bn_gamma,
            beta_layer=bn_beta,
            axis=1,
        )  # 1, 2, 3, 3
        concat = px.ops.concat("concat1", [pool, bn], axis=1)  # 1, 4, 3, 3
        w2 = px.ops.constant("weight2", np.ones((6, 2, 2, 2), dtype=np.float32))
        conv2 = px.ops.conv2d(
            op_name="conv2",
            input_layer=concat,
            weights_layer=w2,
            kernel_size=[2, 2],
            padding_hw=[1, 1, 0, 0],
        )  # 1, 6, 3, 3
        net = [x1, conv1, pool, bn, concat, conv2]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 6
        assert p_xgraph.get_subgraph_names() == ["xp0"]

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Pooling"]
        assert p_xlayers[3].type[0] in ["BatchNorm"]
        assert p_xlayers[4].type[0] in ["Concat"]
        assert p_xlayers[5].type[0] in ["Convolution"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"
        assert p_xlayers[3].target == "cpu"
        assert p_xlayers[4].target == "cpu"
        assert p_xlayers[5].target == "cpu"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp0"
        assert p_xlayers[2].subgraph == "xp0"
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph is None
        assert p_xlayers[5].subgraph is None

        assert p_xlayers[3].name == "bn1"
        assert p_xlayers[3].bottoms == ["conv1"]
        assert p_xlayers[3].tops == ["concat1"]

        assert p_xlayers[4].name == "concat1"
        assert p_xlayers[4].bottoms == ["pool1", "bn1"]
        assert p_xlayers[4].tops == ["conv2"]

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1"]
        assert xp0.tops == ["bn1", "concat1"]
        assert xp0.shapes == [[-1, 2, 3, 3], [-1, 2, 3, 3]]
        assert xp0.sizes == [18, 18]
        assert xp0.attrs["target"] == "test"
        assert xp0.attrs["__bottom_tensors"] == {"xinput0": ["in1"]}
        assert xp0.attrs["orig_bottom_tensors"] == {"xinput0": ["in1"]}
        assert xp0.attrs["__top_tensors"] == {"conv1": ["bn1"], "pool1": ["concat1"]}
        assert xp0.attrs["orig_top_tensors"] == {"conv1": ["bn1"], "pool1": ["concat1"]}

        assert len(xp0_xgraph) == 3
        xp0_layers = xp0_xgraph.get_layers()

        assert [X.name for X in xp0_xgraph.get_input_layers()] == ["xinput0"]
        # TODO: XGraph only recognizes output layers when they have no top
        #   layers
        assert [X.name for X in xp0_xgraph.get_output_layers()] == ["pool1"]

        assert xp0_layers[0].type[0] == "Input"
        assert xp0_layers[0].layer[0] == "conv1"
        assert xp0_layers[1].type[0] == "Convolution"
        assert xp0_layers[2].type[0] == "Pooling"

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ["conv1"]
        assert xp0_layers[1].bottoms == ["xinput0"]
        assert xp0_layers[1].tops == ["pool1"]
        assert xp0_layers[2].bottoms == ["conv1"]
        assert xp0_layers[2].tops == []

    def test_multiple_partitions(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        x2 = px.ops.input("in2", shape=[1, 2, 2, 2])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=x1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 2, 3, 3
        pool = px.ops.pool2d(
            op_name="pool1", input_layer=conv1, pool_type="Avg", pool_size=[2, 2],
        )  # 1, 2, 2, 2
        add = px.ops.eltwise("add1", pool, x2)  # 1, 2, 2, 2
        bn_mean = px.ops.constant("mean", np.ones((2,), dtype=np.float32))
        bn_var = px.ops.constant("var", np.ones((2,), dtype=np.float32))
        bn_gamma = px.ops.constant("gamma", np.ones((2,), dtype=np.float32))
        bn_beta = px.ops.constant("beta", np.ones((2,), dtype=np.float32))
        bn = px.ops.batch_norm(
            op_name="bn1",
            input_layer=add,
            mean_layer=bn_mean,
            variance_layer=bn_var,
            gamma_layer=bn_gamma,
            beta_layer=bn_beta,
            axis=1,
        )  # 1, 2, 3, 3
        pool2 = px.ops.pool2d(
            op_name="pool2",
            input_layer=bn,
            pool_type="Avg",
            pool_size=[2, 2],
            padding=[0, 0, 1, 1],
        )  # 1, 2, 2, 2
        net = [x1, x2, conv1, pool, add, bn, pool2]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 7
        # ! Only xp0 because only one subgraph can exist for now (largest)
        assert set(p_xgraph.get_subgraph_names()) == set(["xp0"])

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Pooling"]
        assert p_xlayers[3].type[0] in ["Input"]
        assert p_xlayers[4].type[0] in ["Eltwise"]
        assert p_xlayers[5].type[0] in ["BatchNorm"]
        assert p_xlayers[6].type[0] in ["Pooling"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"
        assert p_xlayers[3].target == "cpu"
        assert p_xlayers[4].target == "test"
        assert p_xlayers[5].target == "cpu"
        # ! CPU because only one subgraph can exist for now (largest)
        assert p_xlayers[6].target == "cpu"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp0"
        assert p_xlayers[2].subgraph == "xp0"
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph == "xp0"
        assert p_xlayers[5].subgraph is None
        assert p_xlayers[6].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1", "in2"]
        assert xp0.tops == ["bn1"]
        assert xp0.shapes == [[-1, 2, 2, 2]]
        assert xp0.sizes == [8]

        assert len(xp0_xgraph) == 5
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == "Input"
        assert xp0_layers[0].layer[0] == "conv1"
        assert xp0_layers[1].type[0] == "Convolution"
        assert xp0_layers[2].type[0] == "Pooling"
        assert xp0_layers[3].type[0] == "Input"
        assert xp0_layers[4].type[0] == "Eltwise"

        assert xp0_layers[0].bottoms == []
        assert xp0_layers[0].tops == ["conv1"]
        assert xp0_layers[1].bottoms == ["xinput0"]
        assert xp0_layers[1].tops == ["pool1"]
        assert xp0_layers[2].bottoms == ["conv1"]
        assert xp0_layers[2].tops == ["add1"]

    def test_multiple_partitions_largest_last(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        w1 = px.ops.constant("weight1", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=x1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 2, 3, 3
        t1 = px.ops.transpose("t1", conv1, axes=[0, 2, 3, 1])  # 1, 3, 3, 2
        w2 = px.ops.constant("weight2", np.ones((4, 2, 2, 2), dtype=np.float32))
        conv2 = px.ops.conv2d(
            op_name="conv2",
            input_layer=t1,
            weights_layer=w2,
            kernel_size=[2, 2],
            padding_hw=[1, 0, 1, 0],
            data_layout="NHWC",
        )  # 1, 3, 3, 4
        pool = px.ops.pool2d(
            op_name="pool1",
            input_layer=conv2,
            pool_type="Avg",
            pool_size=[2, 2],
            layout="NHWC",
        )  # 1, 2, 2, 4
        net = [x1, conv1, t1, conv2, pool]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 5
        # ! Only xp1 because only one subgraph can exist for now (largest)
        assert set(p_xgraph.get_subgraph_names()) == set(["xp1"])

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Transpose"]
        assert p_xlayers[3].type[0] in ["Convolution"]
        assert p_xlayers[4].type[0] in ["Pooling"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "cpu"
        assert p_xlayers[2].target == "cpu"
        assert p_xlayers[3].target == "test"
        assert p_xlayers[4].target == "test"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph is None
        assert p_xlayers[2].subgraph is None
        assert p_xlayers[3].subgraph == "xp1"
        assert p_xlayers[4].subgraph == "xp1"

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp1 = subgraphs[0]
        assert xp1.name == "xp1"
        xp1_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp1.subgraph_data
        )

        assert xp1.bottoms == ["t1"]
        assert xp1.tops == []
        assert xp1.shapes == [[-1, 2, 2, 4]]
        assert xp1.sizes == [16]

        assert len(xp1_xgraph) == 3
        xp1_layers = xp1_xgraph.get_layers()

        assert xp1_layers[0].type[0] == "Input"
        assert xp1_layers[0].layer[0] == "conv2"
        assert xp1_layers[1].type[0] == "Convolution"
        assert xp1_layers[2].type[0] == "Pooling"

        assert xp1_layers[0].bottoms == []
        assert xp1_layers[0].tops == ["conv2"]
        assert xp1_layers[1].bottoms == ["xinput0"]
        assert xp1_layers[1].tops == ["pool1"]
        assert xp1_layers[2].bottoms == ["conv2"]
        assert xp1_layers[2].tops == []

    def test_two_partition_inputs(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        x2 = px.ops.input("in2", shape=[1, 1, 4, 4])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=x1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 2, 3, 3
        pool = px.ops.pool2d(
            op_name="pool1", input_layer=conv1, pool_type="Avg", pool_size=[3, 3],
        )  # 1, 2, 1, 1
        w2 = px.ops.constant("weight2", np.ones((2, 1, 4, 4), dtype=np.float32))
        conv2 = px.ops.conv2d(
            op_name="conv2",
            input_layer=x2,
            weights_layer=w2,
            kernel_size=[4, 4],
            strides=[1, 1],
        )  # 1, 2, 1, 1
        add = px.ops.eltwise("add1", pool, conv2)
        reshape = px.ops.reshape("reshape1", add, [-1, 2])
        wd = px.ops.constant("weight_dense", np.ones((20, 2), dtype=np.float32))
        dense = px.ops.dense(
            op_name="dense1", input_layer=reshape, weights_layer=wd, units=20,
        )
        net = [x1, x2, conv1, pool, conv2, add, reshape, dense]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 8
        assert p_xgraph.get_subgraph_names() == ["xp2"]

        p_xlayers = p_xgraph.get_layers()

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"
        assert p_xlayers[3].target == "cpu"
        assert p_xlayers[4].target == "test"
        assert p_xlayers[5].target == "test"
        assert p_xlayers[6].target == "cpu"
        assert p_xlayers[7].target == "cpu"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp2"
        assert p_xlayers[2].subgraph == "xp2"
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph == "xp2"
        assert p_xlayers[5].subgraph == "xp2"
        assert p_xlayers[6].subgraph is None
        assert p_xlayers[7].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp2 = subgraphs[0]
        assert xp2.name == "xp2"
        xp2_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp2.subgraph_data
        )

        assert xp2.bottoms == ["in1", "in2"]
        assert xp2.tops == ["reshape1"]
        assert xp2.shapes == [[-1, 2, 1, 1]]
        assert xp2.sizes == [2]

        assert len(xp2_xgraph) == 6
        xp2_layers = xp2_xgraph.get_layers()

        assert xp2_layers[0].type[0] == "Input"
        assert xp2_layers[0].layer[0] == "conv1"
        assert xp2_layers[1].type[0] == "Convolution"
        assert xp2_layers[2].type[0] == "Pooling"
        assert xp2_layers[3].type[0] == "Input"
        assert xp2_layers[3].layer[0] == "conv2"
        assert xp2_layers[4].type[0] == "Convolution"
        assert xp2_layers[5].type[0] == "Eltwise"

        assert xp2_layers[0].bottoms == []
        assert xp2_layers[0].tops == ["conv1"]
        assert xp2_layers[1].bottoms == ["xinput0"]
        assert xp2_layers[1].tops == ["pool1"]
        assert xp2_layers[2].bottoms == ["conv1"]
        assert xp2_layers[2].tops == ["add1"]
        assert xp2_layers[3].bottoms == []
        assert xp2_layers[3].tops == ["conv2"]
        assert xp2_layers[4].bottoms == ["xinput1"]
        assert xp2_layers[4].tops == ["add1"]
        assert xp2_layers[5].bottoms == ["pool1", "conv2"]
        assert xp2_layers[5].tops == []

    def test_inception_like_block(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        x2 = px.ops.input("in2", shape=[1, 1, 4, 4])
        concat1 = px.ops.concat("concat1", [x1, x2], axis=1)  # 1, 2, 4, 4
        w1 = px.ops.constant("weight", np.ones((4, 2, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=concat1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 4, 3, 3
        pool1 = px.ops.pool2d(
            op_name="pool1", input_layer=conv1, pool_type="Avg", pool_size=[2, 2],
        )  # 1, 4, 2, 2
        w2 = px.ops.constant("weight2", np.ones((4, 2, 2, 2), dtype=np.float32))
        conv2 = px.ops.conv2d(
            op_name="conv2",
            input_layer=concat1,
            weights_layer=w2,
            kernel_size=[2, 2],
            strides=[2, 2],
        )  # 1, 4, 2, 2
        concat2 = px.ops.concat("concat2", [pool1, conv2], axis=1)  # 1, 8, 2, 2
        net = [x1, x2, concat1, conv1, pool1, conv2, concat2]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 7
        p_xlayers = p_xgraph.get_layers()

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "cpu"
        assert p_xlayers[2].target == "cpu"
        assert p_xlayers[3].target == "test"
        assert p_xlayers[4].target == "test"
        assert p_xlayers[5].target == "cpu"
        assert p_xlayers[6].target == "cpu"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph is None
        assert p_xlayers[2].subgraph is None
        assert p_xlayers[3].subgraph == "xp0"
        assert p_xlayers[4].subgraph == "xp0"
        assert p_xlayers[5].subgraph is None
        assert p_xlayers[6].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["concat1"]
        assert xp0.tops == ["concat2"]
        assert xp0.shapes == [[-1, 4, 2, 2]]
        assert xp0.sizes == [16]

        assert len(xp0_xgraph) == 3
        xp0_layers = xp0_xgraph.get_layers()

        assert xp0_layers[0].type[0] == "Input"
        assert xp0_layers[1].type[0] == "Convolution"
        assert xp0_layers[2].type[0] == "Pooling"

    def test_top_tensors_basic(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=x1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 2, 3, 3
        pool1 = px.ops.pool2d(
            op_name="pool1", input_layer=conv1, pool_type="Avg", pool_size=[2, 2],
        )  # 1, 2, 2, 2

        # ! internal=1
        t1 = px.ops.transpose("t1", pool1, axes=[0, 2, 3, 1], internal=1)
        s1 = px.ops.sqrt("s1", [t1])
        net = [x1, conv1, pool1, t1, s1]
        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 5
        assert p_xgraph.get_subgraph_names() == ["xp0"]

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Pooling"]
        assert p_xlayers[3].type[0] in ["Transpose"]
        assert p_xlayers[4].type[0] in ["Sqrt"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"
        assert p_xlayers[3].target == "cpu"
        assert p_xlayers[4].target == "cpu"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp0"
        assert p_xlayers[2].subgraph == "xp0"
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1"]
        assert xp0.tops == ["t1"]
        assert xp0.shapes == [[-1, 2, 2, 2]]
        assert xp0.sizes == [8]
        assert len(xp0_xgraph) == 3

        __bottom_tensors = xp0.attrs["__bottom_tensors"]
        orig_bottom_tensors = xp0.attrs["orig_bottom_tensors"]

        assert len(__bottom_tensors) == 1
        assert "xinput0" in __bottom_tensors
        assert __bottom_tensors["xinput0"] == ["in1"]

        assert len(orig_bottom_tensors) == 1
        assert "xinput0" in orig_bottom_tensors
        assert orig_bottom_tensors["xinput0"] == ["in1"]

        __top_tensors = xp0.attrs["__top_tensors"]
        orig_top_tensors = xp0.attrs["orig_top_tensors"]

        assert len(__top_tensors) == 1
        assert "pool1" in __top_tensors
        assert __top_tensors["pool1"] == ["t1"]

        assert len(orig_top_tensors) == 1
        assert "pool1" in orig_top_tensors
        assert orig_top_tensors["pool1"] == ["s1"]

    def test_multi_top_tensors(self):
        x1 = px.ops.input("in1", shape=[1, 1, 4, 4])
        w1 = px.ops.constant("weight", np.ones((2, 1, 2, 2), dtype=np.float32))
        conv1 = px.ops.conv2d(
            op_name="conv1", input_layer=x1, weights_layer=w1, kernel_size=[2, 2],
        )  # 1, 2, 3, 3
        pool1 = px.ops.pool2d(
            op_name="pool1", input_layer=conv1, pool_type="Avg", pool_size=[2, 2],
        )  # 1, 2, 2, 2
        t1 = px.ops.transpose("t1", pool1, axes=[0, 2, 3, 1], internal=1)
        t2 = px.ops.transpose("t2", pool1, axes=[0, 2, 3, 1], internal=1)
        s1 = px.ops.sqrt("s1", [t1])
        s2 = px.ops.sqrt("s2", [t2])
        s3 = px.ops.sqrt("s3", [t2])
        net = [x1, conv1, pool1, t1, t2, s1, s2, s3]

        xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(net)
        p_xgraph = px.partition(xgraph, ["test"])

        assert len(p_xgraph.get_layer_names()) == 8
        assert p_xgraph.get_subgraph_names() == ["xp0"]

        p_xlayers = p_xgraph.get_layers()
        assert p_xlayers[0].type[0] in ["Input"]
        assert p_xlayers[1].type[0] in ["Convolution"]
        assert p_xlayers[2].type[0] in ["Pooling"]
        assert p_xlayers[3].type[0] in ["Transpose"]
        assert p_xlayers[4].type[0] in ["Sqrt"]
        assert p_xlayers[5].type[0] in ["Transpose"]
        assert p_xlayers[6].type[0] in ["Sqrt"]
        assert p_xlayers[7].type[0] in ["Sqrt"]

        assert p_xlayers[0].target == "cpu"
        assert p_xlayers[1].target == "test"
        assert p_xlayers[2].target == "test"
        assert p_xlayers[3].target == "cpu"
        assert p_xlayers[4].target == "cpu"
        assert p_xlayers[5].target == "cpu"
        assert p_xlayers[6].target == "cpu"
        assert p_xlayers[7].target == "cpu"

        assert p_xlayers[0].subgraph is None
        assert p_xlayers[1].subgraph == "xp0"
        assert p_xlayers[2].subgraph == "xp0"
        assert p_xlayers[3].subgraph is None
        assert p_xlayers[4].subgraph is None
        assert p_xlayers[5].subgraph is None
        assert p_xlayers[6].subgraph is None
        assert p_xlayers[7].subgraph is None

        subgraphs = TestXGraphPartitioner.xgraph_partitioner.get_subgraphs(p_xgraph)

        assert len(subgraphs) == 1
        xp0 = subgraphs[0]
        assert xp0.name == "xp0"
        xp0_xgraph = TestXGraphPartitioner.xgraph_factory.build_from_xlayer(
            xp0.subgraph_data
        )

        assert xp0.bottoms == ["in1"]
        assert xp0.tops == ["t1", "t2"]
        assert xp0.shapes == [[-1, 2, 2, 2], [-1, 2, 2, 2]]
        assert xp0.sizes == [8, 8]
        assert len(xp0_xgraph) == 3

        __bottom_tensors = xp0.attrs["__bottom_tensors"]
        orig_bottom_tensors = xp0.attrs["orig_bottom_tensors"]

        assert len(__bottom_tensors) == 1
        assert "xinput0" in __bottom_tensors
        assert __bottom_tensors["xinput0"] == ["in1"]

        assert len(orig_bottom_tensors) == 1
        assert "xinput0" in orig_bottom_tensors
        assert orig_bottom_tensors["xinput0"] == ["in1"]

        __top_tensors = xp0.attrs["__top_tensors"]
        orig_top_tensors = xp0.attrs["orig_top_tensors"]

        assert len(__top_tensors) == 1
        assert "pool1" in __top_tensors
        assert __top_tensors["pool1"] == ["t1", "t2"]

        assert len(orig_top_tensors) == 1
        assert "pool1" in orig_top_tensors
        assert orig_top_tensors["pool1"] == ["s1", "s2", "s3"]


if __name__ == "__main__":
    unittest.main()
