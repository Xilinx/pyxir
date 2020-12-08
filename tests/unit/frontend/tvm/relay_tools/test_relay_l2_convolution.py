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

"""Module for testing the relay pyxir frontend"""

import unittest
import numpy as np

# ! To import tvm
import pyxir

try:
    import tvm
    from tvm import relay
    from tvm.relay import testing

    skip = False
except Exception as e:
    skip = True

if not skip:
    from pyxir.frontend.tvm import relay as xf_relay


class TestRelayL2Convolutions(unittest.TestCase):

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_avg_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 5, 5), "float32"))
        avg_pool = relay.nn.avg_pool2d(var, pool_size=(3, 3), strides=(2, 2), padding=(1, 1),
                                 ceil_mode=True, count_include_pad=True)

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Pooling'
        assert X.shapes == [-1, 2, 3, 3]
        assert 'relay_id' in X.attrs
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs['insize'] == [5, 5]
        assert X.attrs['outsize'] == [3, 3]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['strides'] == [2, 2]
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['pool_type'] == 'Avg'

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_batch_flatten(self):
        data = relay.var("data", relay.TensorType((-1, 1, 1, 4), "float32"))

        net = relay.nn.batch_flatten(data)
        net = relay.Function([data], net)
        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        xgraph = xf_relay.from_relay(mod, {})
        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
    
        assert layers[1].type[0] == 'Flatten'
        assert layers[1].shapes == [-1, 4]
        assert 'relay_id' in layers[1].attrs

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_conv2d(self):
        data = relay.var("data", relay.TensorType((-1, 1, 4, 4), "float32"))
        weight = relay.expr.const(np.ones((2, 1, 2, 2), dtype=np.float32))
        c = relay.nn.conv2d(data, weight, padding=(0, 0, 0, 0), kernel_layout='OIHW')
        
        func = relay.Function([data], c)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Convolution'
        assert X.shapes == [-1, 2, 3, 3]
        np.testing.assert_array_equal(X.data[0], np.ones((2, 1, 2, 2), dtype=np.float32))
        assert 'relay_id' in X.attrs
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['channels'] == [1, 2]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['groups'] == 1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_conv2d_transpose(self):
        data = relay.var("data", relay.TensorType((-1, 2, 3, 3), "float32"))
        weight = relay.expr.const(np.ones((2, 4, 3, 3), dtype=np.float32))
        c = relay.nn.conv2d_transpose(data, weight, padding=(0, 0, 0, 0), kernel_layout='OIHW')
        
        func = relay.Function([data], c)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Conv2DTranspose'
        assert X.shapes == [-1, 4, 5, 5]
        np.testing.assert_array_equal(X.data[0], np.ones((4, 2, 3, 3), dtype=np.float32))
        assert 'relay_id' in X.attrs
        assert X.attrs['kernel_size'] == [3, 3]
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['channels'] == [2, 4]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['kernel_layout'] == 'OIHW'
        assert X.attrs['groups'] == 1

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_global_avg_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 5, 5), "float32"))
        avg_pool = relay.nn.global_avg_pool2d(var)

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Pooling'
        assert X.bottoms == ['var']
        assert X.shapes == [-1, 2, 1, 1]
        assert 'relay_id' in X.attrs
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['insize'] == [5, 5]
        assert X.attrs['outsize'] == [1, 1]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['kernel_size'] == [5, 5]
        assert X.attrs['pool_type'] == 'Avg'

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_global_max_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 5, 5), "float32"))
        avg_pool = relay.nn.global_max_pool2d(var)

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Pooling'
        assert X.bottoms == ['var']
        assert X.shapes == [-1, 2, 1, 1]
        assert 'relay_id' in X.attrs
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 0], [0, 0]]
        assert X.attrs['insize'] == [5, 5]
        assert X.attrs['outsize'] == [1, 1]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['strides'] == [1, 1]
        assert X.attrs['kernel_size'] == [5, 5]
        assert X.attrs['pool_type'] == 'Max'

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_max_pool2d(self):
        var = relay.var("var", relay.TensorType((-1, 2, 4, 4), "float32"))
        avg_pool = relay.nn.max_pool2d(var, pool_size=(2, 2), strides=(2, 2), padding=(1, 1))

        func = relay.Function([var], avg_pool)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Pooling'
        assert X.bottoms == ['var']
        assert X.shapes == [-1, 2, 3, 3]
        assert 'relay_id' in X.attrs
        assert X.attrs['padding'] == [[0, 0], [0, 0], [1, 1], [1, 1]]
        assert X.attrs['insize'] == [4, 4]
        assert X.attrs['outsize'] == [3, 3]
        assert X.attrs['data_layout'] == 'NCHW'
        assert X.attrs['strides'] == [2, 2]
        assert X.attrs['kernel_size'] == [2, 2]
        assert X.attrs['pool_type'] == 'Max'

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_padding(self):
        var = relay.var("var", relay.TensorType((-1, 2, 4, 4), "float32"))
        pad = relay.nn.pad(var, ((0, 0), (0, 0), (0, 1), (0, 1)))

        func = relay.Function([var], pad)
        mod = tvm.IRModule.from_expr(func)
        mod = relay.transform.InferType()(mod)

        xg = xf_relay.from_relay(mod, {})
        layers = xg.get_layers()

        assert len(layers) == 2

        assert layers[0].type[0] == 'Input'
        assert 'relay_id' in layers[0].attrs

        X = layers[1]
        assert X.type[0] == 'Pad'
        assert X.bottoms == ['var']
        assert X.shapes == [-1, 2, 5, 5]
        assert 'relay_id' in X.attrs
        assert X.attrs['padding'] == [[0, 0], [0, 0], [0, 1], [0, 1]]

    @unittest.skipIf(skip, "Could not import TVM and/or TVM frontend")
    def test_nn_upsampling(self):
        data = relay.var(
            "data",
            relay.TensorType((-1, 4, 2, 2), "float32")
        )

        net = relay.nn.upsampling(data, scale_h=3, scale_w=2)

        net = relay.Function([data], net)

        mod = tvm.IRModule.from_expr(net)
        mod = relay.transform.InferType()(mod)

        params = {}

        xgraph = xf_relay.from_relay(mod, params)

        layers = xgraph.get_layers()

        assert layers[0].type[0] == 'Input'
        assert layers[1].type[0] == 'Upsampling2D'
        assert 'relay_id' in layers[1].attrs
        assert layers[1].shapes == [-1, 4, 6, 4]
        assert layers[1].attrs['scale_h'] == 3
        assert layers[1].attrs['scale_w'] == 2
        assert layers[1].attrs['data_layout'] == 'NCHW'
        assert layers[1].attrs['method'] == 'nearest_neighbor'
        assert layers[1].attrs['align_corners'] is False
