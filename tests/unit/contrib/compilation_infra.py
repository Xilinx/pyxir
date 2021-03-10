# Copyright 2021 Xilinx Inc.
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

"""Utilities for testing DPUCZDX8G compilation"""

import os
import xir
import shutil
import numpy as np
import pyxir as px

from pyxir.target_registry import TargetRegistry
from pyxir.graph import XGraph
from pyxir.graph.xgraph_factory import XGraphFactory

XGRAPH_FACTORY = XGraphFactory()
TARGET_REGISTRY = TargetRegistry()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def remove_all_files_with_suffix(dir_path, suffix):
    files_with_suffix = [f for f in os.listdir(dir_path) if f.endswith(suffix)]
    [os.remove(os.path.join(FILE_PATH, f)) for f in files_with_suffix]


def get_child_subgraphs(graph: "Graph"):
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    # import pdb; pdb.set_trace()
    return child_subgraphs
    # return [
    #     cs
    #     for cs in child_subgraphs
    #     if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    # ]


def _create_conv2d_pool2d_nhwc_oihw(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    pool_type,
    pool_size,
    pool_padding=[0, 0],
    pool_strides=[1, 1],
    conv_groups=1,
    conv_invalid=False,
    kernel_layout="OIHW",
    target="DPUCZDX8G-zcu104",
) -> XGraph:

    kernel_w, kernel_h = w_shape[2], w_shape[3]
    W = np.random.randint(-10, 10, size=w_shape).astype(np.float32)
    # B = np.array([1., -1.], dtype=np.float32)

    x1 = px.ops.input("in1", shape=list(in_shape))
    w1 = px.ops.constant("weight", W)
    conv1 = px.ops.conv2d(
        op_name="conv1",
        input_layer=x1,
        weights_layer=w1,
        kernel_size=[kernel_w, kernel_h],
        strides=list(conv_strides),
        padding_hw=list(conv_padding),
        dilation=list(conv_dilation),
        groups=conv_groups,
        data_layout="NHWC",
    )
    pool1 = px.ops.pool2d(
        op_name="pool1",
        input_layer=conv1,
        pool_type=pool_type,
        pool_size=list(pool_size),
        padding=list(pool_padding),
        layout="NHWC",
    )
    net = [x1, conv1, pool1]
    xgraph = XGRAPH_FACTORY.build_from_xlayer(net)
    xgraph = px.partition(xgraph, [target])
    return xgraph


def conv2d_pool2d_nhwc_oihw_test(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    pool_type,
    pool_size,
    pool_padding=[0, 0],
    pool_strides=[1, 1],
    conv_groups=1,
    conv_invalid=False,
    kernel_layout="OIHW",
    targets=["DPUCZDX8G-zcu104"],
) -> None:

    for target in targets:
        xgraph = _create_conv2d_pool2d_nhwc_oihw(
            in_shape,
            w_shape,
            conv_padding,
            conv_strides,
            conv_dilation,
            pool_type,
            pool_size,
            pool_padding,
            pool_strides,
            conv_groups,
            conv_invalid,
            kernel_layout,
            target,
        )

        def inputs_func(iter):
            inputs = np.ones(in_shape, dtype=np.float32)
            return {"in1": inputs}

        work_dir = os.path.join(FILE_PATH, "work")
        build_dir = os.path.join(FILE_PATH, "build")
        quantize_func = TARGET_REGISTRY.get_target_quantizer(target)
        q_xgraph = quantize_func(xgraph, inputs_func, work_dir=work_dir)
        opt_xgraph = px.optimize(q_xgraph, target)
        c_xgraph = px.compile(
            opt_xgraph, target, work_dir=work_dir, build_dir=build_dir
        )
        c_output = c_xgraph.get_compiler_output()

        assert list(c_output.keys()) == ["xp0"]
        assert c_output.get_in_map("xp0") == {"xinput0": "xinput0:0"}
        assert c_output.get_out_map("xp0") == {"pool1": "pool1:0"}
        assert len(c_output.get_code_files("xp0")) == 1

        shutil.rmtree(work_dir)
        shutil.rmtree(build_dir)


def xcompiler_conv2d_pool2d_nhwc_oihw_test(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    pool_type,
    pool_size,
    pool_padding=[0, 0],
    pool_strides=[1, 1],
    conv_groups=1,
    conv_invalid=False,
    kernel_layout="OIHW",
    targets=["DPUCAHX8H-u50"],
    expected_nb_subgraphs=3,
):

    for target in targets:
        xgraph = _create_conv2d_pool2d_nhwc_oihw(
            in_shape,
            w_shape,
            conv_padding,
            conv_strides,
            conv_dilation,
            pool_type,
            pool_size,
            pool_padding,
            pool_strides,
            conv_groups,
            conv_invalid,
            kernel_layout,
            target,
        )

        def inputs_func(iter):
            inputs = np.ones(in_shape, dtype=np.float32)
            return {"in1": inputs}

        work_dir = os.path.join(FILE_PATH, "work")
        build_dir = os.path.join(FILE_PATH, "build")
        quantize_func = TARGET_REGISTRY.get_target_quantizer(target)
        q_xgraph = quantize_func(xgraph, inputs_func, work_dir=work_dir)
        opt_xgraph = px.optimize(q_xgraph, target)
        c_xgraph = px.compile(
            opt_xgraph, target, work_dir=work_dir, build_dir=build_dir
        )
        c_output = c_xgraph.get_compiler_output()

        assert list(c_output.keys()) == ["xp0"]
        assert c_output.get_in_map("xp0") == {"xinput0": "xinput0"}
        assert c_output.get_out_map("xp0") == {"pool1": "pool1"}
        assert len(c_output.get_code_files("xp0")) == 1

        g = xir.Graph.deserialize(os.path.join(build_dir, "xp0.xmodel"))
        # TODO subgraphs[1].get_attr("device") -> *** RuntimeError: bad any_cast
        subgraphs = get_child_subgraphs(g)
        assert len(subgraphs) == expected_nb_subgraphs
        dpu_subgraph = subgraphs[1]
        # import pdb; pdb.set_trace()
        # assert len(dpu_subgraph.get_children()) == 3

        shutil.rmtree(work_dir)
        shutil.rmtree(build_dir)


def _create_scale_conv2d_nhwc_oihw(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    conv_groups=1,
    kernel_layout="OIHW",
    target="DPUCZDX8G-zcu104",
) -> XGraph:

    kernel_w, kernel_h = w_shape[2], w_shape[3]
    in_ch = w_shape[0]

    Gamma = np.random.randint(0, 2, size=(in_ch,))
    Beta = np.random.randint(0, 2, size=(in_ch,))
    W = np.random.randint(-10, 10, size=w_shape).astype(np.float32)
    # B = np.array([1., -1.], dtype=np.float32)

    x1 = px.ops.input("in1", shape=list(in_shape))
    w1 = px.ops.constant("weight", W)
    conv1 = px.ops.conv2d(
        op_name="conv1",
        input_layer=x1,
        weights_layer=w1,
        kernel_size=[kernel_w, kernel_h],
        strides=list(conv_strides),
        padding_hw=list(conv_padding),
        dilation=list(conv_dilation),
        groups=conv_groups,
        data_layout="NHWC",
    )

    pool1 = px.ops.pool2d(
        op_name="pool1",
        input_layer=conv1,
        pool_type="Max",
        pool_size=[3, 3],
        padding=[0, 0],
        layout="NHWC",
    )

    g1 = px.ops.constant("gamma", Gamma)
    b1 = px.ops.constant("beta", Beta)
    scale = px.ops.scale("scale1", pool1, g1, b1, axis=3)
    r1 = px.ops.relu("r1", [scale])

    W2 = np.random.randint(-10, 10, size=(in_ch, in_ch, 1, 1)).astype(np.float32)
    w2 = px.ops.constant("weight2", W2)
    conv2 = px.ops.conv2d(
        op_name="conv2",
        input_layer=r1,
        weights_layer=w2,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding_hw=[0, 0],
        dilation=[1, 1],
        groups=1,
        data_layout="NHWC",
    )

    net = [x1, conv1, pool1, scale, r1, conv2]
    xgraph = XGRAPH_FACTORY.build_from_xlayer(net)
    xgraph = px.partition(xgraph, [target])
    return xgraph


def xcompiler_scale_conv2d_nhwc_oihw_test(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    conv_groups=1,
    kernel_layout="OIHW",
    target="DPUCAHX8H-u50",
    expected_nb_subgraphs=3,
):

    xgraph = _create_scale_conv2d_nhwc_oihw(
        in_shape,
        w_shape,
        conv_padding,
        conv_strides,
        conv_dilation,
        conv_groups,
        kernel_layout,
        target,
    )

    def inputs_func(iter):
        inputs = np.ones(in_shape, dtype=np.float32)
        return {"in1": inputs}

    work_dir = os.path.join(FILE_PATH, "work")
    build_dir = os.path.join(FILE_PATH, "build")
    quantize_func = TARGET_REGISTRY.get_target_quantizer(target)
    q_xgraph = quantize_func(xgraph, inputs_func, work_dir=work_dir)
    opt_xgraph = px.optimize(q_xgraph, target)
    c_xgraph = px.compile(opt_xgraph, target, work_dir=work_dir, build_dir=build_dir)
    c_output = c_xgraph.get_compiler_output()

    g = xir.Graph.deserialize(os.path.join(build_dir, "xp0.xmodel"))
    # TODO subgraphs[1].get_attr("device") -> *** RuntimeError: bad any_cast
    subgraphs = get_child_subgraphs(g)
    assert (
        len(subgraphs) == expected_nb_subgraphs
    ), "Expected {0} subgraphs but got: {1}".format(
        expected_nb_subgraphs, len(subgraphs)
    )

    shutil.rmtree(work_dir)
    shutil.rmtree(build_dir)


def _create_resnetv1_block(
    in_shape,
    pool_size,
    pool_strides,
    w1_shape,
    w2_shape,
    w3_shape,
    w4_shape,
    c1_padding=[0, 0, 0, 0],
    c2_padding=[0, 0, 0, 0],
    c3_padding=[0, 0, 0, 0],
    c4_padding=[0, 0, 0, 0],
    c1_strides=[1, 1],
    c2_strides=[1, 1],
    c3_strides=[1, 1],
    c4_strides=[1, 1],
    c1_dilation=[1, 1],
    c2_dilation=[1, 1],
    c3_dilation=[1, 1],
    c4_dilation=[1, 1],
    kernel_layout="OIHW",
    target="DPUCZDX8G-zcu104",
) -> XGraph:

    x1 = px.ops.input("in1", shape=list(in_shape))
    pool1 = px.ops.pool2d(
        op_name="pool1",
        input_layer=x1,
        pool_type="Max",
        pool_size=pool_size,
        padding=[0, 0],
        strides=pool_strides,
        layout="NHWC",
    )

    W1 = np.random.randint(-10, 10, size=w1_shape).astype(np.float32)
    w1 = px.ops.constant("w1", W1)
    conv1 = px.ops.conv2d(
        op_name="conv1",
        input_layer=pool1,
        weights_layer=w1,
        kernel_size=[w1_shape[2], w1_shape[3]],
        strides=list(c1_strides),
        padding_hw=list(c1_padding),
        dilation=list(c1_dilation),
        groups=1,
        data_layout="NHWC",
    )

    W2 = np.random.randint(-10, 10, size=w2_shape).astype(np.float32)
    w2 = px.ops.constant("w2", W2)
    conv2 = px.ops.conv2d(
        op_name="conv2",
        input_layer=pool1,
        weights_layer=w2,
        kernel_size=[w2_shape[2], w2_shape[3]],
        strides=list(c2_strides),
        padding_hw=list(c2_padding),
        dilation=list(c2_dilation),
        groups=1,
        data_layout="NHWC",
    )

    W3 = np.random.randint(-10, 10, size=w3_shape).astype(np.float32)
    w3 = px.ops.constant("w3", W3)
    conv3 = px.ops.conv2d(
        op_name="conv3",
        input_layer=conv2,
        weights_layer=w3,
        kernel_size=[w3_shape[2], w3_shape[3]],
        strides=list(c3_strides),
        padding_hw=list(c3_padding),
        dilation=list(c3_dilation),
        groups=1,
        data_layout="NHWC",
    )

    W4 = np.random.randint(-10, 10, size=w4_shape).astype(np.float32)
    w4 = px.ops.constant("w4", W4)
    conv4 = px.ops.conv2d(
        op_name="conv4",
        input_layer=conv3,
        weights_layer=w4,
        kernel_size=[w4_shape[2], w4_shape[3]],
        strides=list(c4_strides),
        padding_hw=list(c4_padding),
        dilation=list(c4_dilation),
        groups=1,
        data_layout="NHWC",
    )

    add = px.ops.eltwise("add", conv1, conv4)

    net = [x1, pool1, conv1, conv2, conv3, conv4, add]
    xgraph = XGRAPH_FACTORY.build_from_xlayer(net)
    xgraph = px.partition(xgraph, [target])
    return xgraph


def xcompiler_resnetv1_block_test(
    in_shape,
    pool_size,
    pool_strides,
    w1_shape,
    w2_shape,
    w3_shape,
    w4_shape,
    c1_padding=[0, 0, 0, 0],
    c2_padding=[0, 0, 0, 0],
    c3_padding=[0, 0, 0, 0],
    c4_padding=[0, 0, 0, 0],
    c1_strides=[1, 1],
    c2_strides=[1, 1],
    c3_strides=[1, 1],
    c4_strides=[1, 1],
    c1_dilation=[1, 1],
    c2_dilation=[1, 1],
    c3_dilation=[1, 1],
    c4_dilation=[1, 1],
    kernel_layout="OIHW",
    target="DPUCAHX8H-u50",
    expected_nb_subgraphs=3,
):

    xgraph = _create_resnetv1_block(
        in_shape,
        pool_size,
        pool_strides,
        w1_shape,
        w2_shape,
        w3_shape,
        w4_shape,
        c1_padding,
        c2_padding,
        c3_padding,
        c4_padding,
        c1_strides,
        c2_strides,
        c3_strides,
        c4_strides,
        c1_dilation,
        c2_dilation,
        c3_dilation,
        c4_dilation,
        kernel_layout,
        target,
    )

    def inputs_func(iter):
        inputs = np.ones(in_shape, dtype=np.float32)
        return {"in1": inputs}

    work_dir = os.path.join(FILE_PATH, "work")
    build_dir = os.path.join(FILE_PATH, "build")
    quantize_func = TARGET_REGISTRY.get_target_quantizer(target)
    q_xgraph = quantize_func(xgraph, inputs_func, work_dir=work_dir)
    opt_xgraph = px.optimize(q_xgraph, target)
    c_xgraph = px.compile(opt_xgraph, target, work_dir=work_dir, build_dir=build_dir)
    c_output = c_xgraph.get_compiler_output()

    g = xir.Graph.deserialize(os.path.join(build_dir, "xp0.xmodel"))
    # TODO subgraphs[1].get_attr("device") -> *** RuntimeError: bad any_cast
    subgraphs = get_child_subgraphs(g)
    assert (
        len(subgraphs) == expected_nb_subgraphs
    ), "Expected {0} subgraphs but got: {1}".format(
        expected_nb_subgraphs, len(subgraphs)
    )

    shutil.rmtree(work_dir)
    shutil.rmtree(build_dir)


def _create_conv2d_leaky_relu_nhwc_oihw(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    kernel_layout="OIHW",
    target="DPUCZDX8G-zcu104",
) -> XGraph:

    kernel_w, kernel_h = w_shape[2], w_shape[3]
    W = np.random.randint(-10, 10, size=w_shape).astype(np.float32)
    # B = np.array([1., -1.], dtype=np.float32)

    x1 = px.ops.input("in1", shape=list(in_shape))
    w1 = px.ops.constant("weight", W)
    conv1 = px.ops.conv2d(
        op_name="conv1",
        input_layer=x1,
        weights_layer=w1,
        kernel_size=[kernel_w, kernel_h],
        strides=list(conv_strides),
        padding_hw=list(conv_padding),
        dilation=list(conv_dilation),
        data_layout="NHWC",
    )
    lr1 = px.ops.leaky_relu("lr1", [conv1], alpha=0.1)
    net = [x1, conv1, lr1]
    xgraph = XGRAPH_FACTORY.build_from_xlayer(net)
    xgraph = px.partition(xgraph, [target])
    return xgraph


def conv2d_leaky_relu_nhwc_oihw_test(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    kernel_layout="OIHW",
    targets=["DPUCZDX8G-zcu104"],
) -> None:

    for target in targets:
        xgraph = _create_conv2d_leaky_relu_nhwc_oihw(
            in_shape,
            w_shape,
            conv_padding,
            conv_strides,
            conv_dilation,
            kernel_layout,
            target,
        )

        def inputs_func(iter):
            inputs = np.ones(in_shape, dtype=np.float32)
            return {"in1": inputs}

        work_dir = os.path.join(FILE_PATH, "work")
        build_dir = os.path.join(FILE_PATH, "build")
        quantize_func = TARGET_REGISTRY.get_target_quantizer(target)
        q_xgraph = quantize_func(xgraph, inputs_func, work_dir=work_dir)
        opt_xgraph = px.optimize(q_xgraph, target)
        c_xgraph = px.compile(
            opt_xgraph, target, work_dir=work_dir, build_dir=build_dir
        )
        c_output = c_xgraph.get_compiler_output()

        assert list(c_output.keys()) == ["xp0"]
        assert c_output.get_in_map("xp0") == {"xinput0": "xinput0:0"}
        assert c_output.get_out_map("xp0") == {"lr1": "lr1:0"}
        assert len(c_output.get_code_files("xp0")) == 1

        shutil.rmtree(work_dir)
        shutil.rmtree(build_dir)


def xcompiler_conv2d_leaky_relu_nhwc_oihw_test(
    in_shape,
    w_shape,
    conv_padding,
    conv_strides,
    conv_dilation,
    kernel_layout="OIHW",
    targets=["DPUCZDX8G-zcu104"],
    expected_nb_subgraphs=3,
) -> None:

    for target in targets:
        xgraph = _create_conv2d_leaky_relu_nhwc_oihw(
            in_shape,
            w_shape,
            conv_padding,
            conv_strides,
            conv_dilation,
            kernel_layout,
            target,
        )

        def inputs_func(iter):
            inputs = np.ones(in_shape, dtype=np.float32)
            return {"in1": inputs}

        work_dir = os.path.join(FILE_PATH, "work")
        build_dir = os.path.join(FILE_PATH, "build")
        quantize_func = TARGET_REGISTRY.get_target_quantizer(target)
        q_xgraph = quantize_func(xgraph, inputs_func, work_dir=work_dir)
        opt_xgraph = px.optimize(q_xgraph, target)
        c_xgraph = px.compile(
            opt_xgraph, target, work_dir=work_dir, build_dir=build_dir
        )
        
        g = xir.Graph.deserialize(os.path.join(build_dir, "xp0.xmodel"))
        # TODO subgraphs[1].get_attr("device") -> *** RuntimeError: bad any_cast
        subgraphs = get_child_subgraphs(g)
        assert (
            len(subgraphs) == expected_nb_subgraphs
        ), "Expected {0} subgraphs but got: {1}".format(
            expected_nb_subgraphs, len(subgraphs)
        )