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

"""Utilities for testing decent quantizer"""

import os
import numpy as np
import pyxir as px

from pyxir.graph.xgraph_factory import XGraphFactory
from pyxir.quantization.decent_quantizer import DECENTQuantizer

XGRAPH_FACTORY = XGraphFactory()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def remove_all_files_with_suffix(dir_path, suffix):
    files_with_suffix = [f for f in os.listdir(dir_path) if f.endswith(suffix)]
    [os.remove(os.path.join(FILE_PATH, f)) for f in files_with_suffix]


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
    target="test-DPU",
):

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

    def inputs_func(iter):
        inputs = np.ones(in_shape, dtype=np.float32)
        return {'in1': inputs}

    quantizer = DECENTQuantizer(xgraph, inputs_func, work_dir=FILE_PATH)
    q_xgraph = quantizer.quantize()
    

    assert len(q_xgraph) == 3
    conv, pool = q_xgraph.get("conv1"), q_xgraph.get("pool1")

    #if not conv_invalid:
    #    assert "vai_quant_weights" in conv.attrs
    #    assert "vai_quant_in" in conv.attrs
    #    assert "vai_quant_out" in conv.attrs
    #    assert "vai_quant" in conv.attrs
    #
    #assert "vai_quant_in" in pool.attrs
    #assert "vai_quant_out" in pool.attrs
    #assert "vai_quant" in pool.attrs

    remove_all_files_with_suffix(FILE_PATH, ".pb")
    remove_all_files_with_suffix(FILE_PATH, ".txt")
