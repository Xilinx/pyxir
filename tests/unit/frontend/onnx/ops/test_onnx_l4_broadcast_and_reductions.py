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
Module for testing the L4 operators for the ONNX frontend


"""

import math
import onnx
import unittest
import numpy as np

from pyxir.graph.layer import xlayer_factory as xlf
from pyxir.frontend.onnx.onnx_tools import NodeWrapper
from pyxir.frontend.onnx.ops import onnx_l4_broadcast_and_reductions as ol4

from pyxir.shapes import TensorShape, TupleShape


class TestONNXL4BroadcastAndReductions(unittest.TestCase):

    def test_broadcast_ops(self):

        any_ops = ['Less', 'LessOrEqual']

        for any_op in any_ops:
            a = np.zeros((1, 2, 3, 3), dtype=np.float32)
            b = np.zeros((3), dtype=np.float32)

            node = onnx.helper.make_node(
                any_op,
                inputs=['a', 'b'],
                outputs=['y']
            )

            wrapped_node = NodeWrapper(node)

            aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                                   dtype='float32')
            bX = xlf.get_xop_factory_func('Constant')('b', b)

            xmap = {'a': aX, 'b': bX}
            params = {}

            func = getattr(ol4, any_op.lower())
            Xs = func(wrapped_node, params, xmap)

            assert len(Xs) == 1
            X = Xs[0]

            assert X.name == 'y'
            assert 'AnyOp' in X.type
            assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_argmax(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'ArgMax',
            inputs=['a'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol4.argmax(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_argmax_keepdims_fals(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'ArgMax',
            inputs=['a'],
            outputs=['y'],
            axis=1,
            keepdims=0
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol4.argmax(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 3, 3]

    def test_argmin(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'ArgMin',
            inputs=['a'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol4.argmin(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_argmin_keepdims_fals(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'ArgMin',
            inputs=['a'],
            outputs=['y'],
            axis=1,
            keepdims=0
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol4.argmin(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 3, 3]

    def test_compress(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        c = np.array([False, True])

        node = onnx.helper.make_node(
            'Compress',
            inputs=['a', 'c'],
            outputs=['y'],
            axis=2
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        cX = xlf.get_xop_factory_func('Constant')('c', c)

        xmap = {'a': aX, 'c': cX}
        params = {}

        Xs = ol4.compress(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        # assert np.compress(c, a, axis=2) == (1, 2, 1, 3)

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 1, 3]

    def test_depth_to_space(self):
        a = np.zeros((1, 8, 3, 3), dtype=np.float32)
        c = np.array([False, True])

        node = onnx.helper.make_node(
            'DepthToSpace',
            inputs=['a'],
            outputs=['y'],
            blocksize=2
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol4.depth_to_space(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        # assert np.compress(c, a, axis=2) == (1, 2, 1, 3)

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 6, 6]

    def test_equal(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Equal',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol4.equal(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_expand(self):
        a = np.zeros((1, 3, 3), dtype=np.float32)
        b = np.array([-1, 1, 3, 3])

        node = onnx.helper.make_node(
            'Expand',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol4.expand(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 1, 3, 3]

    def test_gather(self):
        a = np.zeros((1, 5, 3, 3), dtype=np.float32)
        b = np.array([0, 1, 3])

        node = onnx.helper.make_node(
            'Gather',
            inputs=['a', 'b'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol4.gather(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'Take' in X.type
        assert X.shapes.tolist() == [-1, 3, 3, 3]

    def test_gather_elements(self):
        a = np.zeros((3, 3), dtype=np.float32)
        b = np.zeros((3, 2))

        node = onnx.helper.make_node(
            'GatherElements',
            inputs=['a', 'b'],
            outputs=['y'],
            axis=1
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol4.gather_elements(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [3, 2]

    # def test_gather_nd(self):
    #     a = np.zeros((2, 2, 2), dtype=np.float32)
    #     b = np.array([[0, 1], [1, 0]])

    #     node = onnx.helper.make_node(
    #         'GatherND',
    #         inputs=['a', 'b'],
    #         outputs=['y']
    #     )

    #     wrapped_node = NodeWrapper(node)

    #     aX = xlf.get_xop_factory_func('Constant')('a', a)
    #     bX = xlf.get_xop_factory_func('Constant')('b', b)

    #     xmap = {'a': aX, 'b': bX}
    #     params = {}

    #     Xs = ol4.gather_nd(wrapped_node, params, xmap)

    #     assert len(Xs) == 1
    #     X = Xs[0]

    #     assert X.name == 'y'
    #     assert 'AnyOp' in X.type
    #     assert X.shapes.tolist() == [2, 2]

    def test_greater(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Greater',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol4.greater(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_greater_or_equal(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)

        node = onnx.helper.make_node(
            'GreaterOrEqual',
            inputs=['a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'a': aX, 'b': bX}
        params = {}

        Xs = ol4.greater_or_equal(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_max(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)
        c = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Max',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)
        cX = xlf.get_xop_factory_func('Input')('c', list(c.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX, 'c': cX}
        params = {}

        Xs = ol4.max(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_mean(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)
        c = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Mean',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)
        cX = xlf.get_xop_factory_func('Input')('c', list(c.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX, 'c': cX}
        params = {}

        Xs = ol4.mean(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_min(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        b = np.zeros((3), dtype=np.float32)
        c = np.zeros((1, 2, 3, 3), dtype=np.float32)

        node = onnx.helper.make_node(
            'Min',
            inputs=['a', 'b', 'c'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        bX = xlf.get_xop_factory_func('Constant')('b', b)
        cX = xlf.get_xop_factory_func('Input')('c', list(c.shape),
                                               dtype='float32')

        xmap = {'a': aX, 'b': bX, 'c': cX}
        params = {}

        Xs = ol4.min(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_reduce_to_any_op(self):
        ops = ['ReduceL1', 'ReduceL2', 'ReduceLogSum', 'ReduceMax',
               'ReduceMean', 'ReduceMin', 'ReduceProd', 'ReduceSum',
               'ReduceSumSquare']

        for op in ops:
            a = np.zeros((1, 2, 3, 3), dtype=np.float32)

            node = onnx.helper.make_node(
                op,
                inputs=['a', 'b', 'c'],
                outputs=['y'],
                axes=[2, 3]
            )

            wrapped_node = NodeWrapper(node)

            aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                                   dtype='float32')

            xmap = {'a': aX}
            params = {}

            func = getattr(ol4, op.lower())
            Xs = func(wrapped_node, params, xmap)

            assert len(Xs) == 1
            X = Xs[0]

            assert X.name == 'y'
            assert 'AnyOp' in X.type
            assert X.shapes.tolist() == [-1, 2, 1, 1]

    def test_resize(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        roi = np.array([0, 0, 0, 0, 1, 1, 0.5, 0.5])
        scales = np.array([1, 1, 2, 3])

        node = onnx.helper.make_node(
            'Resize',
            inputs=['a', 'roi', 'scales'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        rX = xlf.get_xop_factory_func('Constant')('roi', roi)
        sX = xlf.get_xop_factory_func('Constant')('scales', scales)

        xmap = {'a': aX, 'roi': rX, 'scales': sX}
        params = {}

        Xs = ol4.resize(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 4]

    def test_roi_align(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        rois = np.zeros((3, 4))
        batch_indices = np.zeros((3))

        node = onnx.helper.make_node(
            'RoiAlign',
            inputs=['a', 'rois', 'batch_indices'],
            outputs=['y'],
            output_height=2,
            output_width=2
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        rX = xlf.get_xop_factory_func('Constant')('rois', rois)
        bX = xlf.get_xop_factory_func('Constant')('batch_indices',
                                                  batch_indices)

        xmap = {'a': aX, 'rois': rX, 'batch_indices': bX}
        params = {}

        Xs = ol4.roi_align(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [3, 2, 2, 2]

    def test_scatter(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        indices = np.zeros((2, 3))
        updates = np.zeros((2, 3))

        node = onnx.helper.make_node(
            'Scatter',
            inputs=['a', 'indices', 'updates'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        iX = xlf.get_xop_factory_func('Constant')('indices', indices)
        uX = xlf.get_xop_factory_func('Constant')('updates', updates)

        xmap = {'a': aX, 'indices': iX, 'updates': uX}
        params = {}

        Xs = ol4.scatter(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_scatter_elements(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        indices = np.zeros((2, 3))
        updates = np.zeros((2, 3))

        node = onnx.helper.make_node(
            'ScatterElements',
            inputs=['a', 'indices', 'updates'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        iX = xlf.get_xop_factory_func('Constant')('indices', indices)
        uX = xlf.get_xop_factory_func('Constant')('updates', updates)

        xmap = {'a': aX, 'indices': iX, 'updates': uX}
        params = {}

        Xs = ol4.scatter_elements(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_scatter_nd(self):
        a = np.zeros((1, 2, 3, 3), dtype=np.float32)
        indices = np.zeros((2, 3))
        updates = np.zeros((2, 3))

        node = onnx.helper.make_node(
            'ScatterND',
            inputs=['a', 'indices', 'updates'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        iX = xlf.get_xop_factory_func('Constant')('indices', indices)
        uX = xlf.get_xop_factory_func('Constant')('updates', updates)

        xmap = {'a': aX, 'indices': iX, 'updates': uX}
        params = {}

        Xs = ol4.scatter_nd(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 3, 3]

    def test_slice(self):
        a = np.zeros((1, 2, 5, 5), dtype=np.float32)
        axes = np.array([2, 3])
        starts = np.array([0, 1])
        ends = np.array([4, 4])
        steps = np.array([2, 2])

        node = onnx.helper.make_node(
            'Slice',
            inputs=['a', 'starts', 'ends', 'axes', 'steps'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')
        starts_X = xlf.get_xop_factory_func('Constant')('starts', starts)
        ends_X = xlf.get_xop_factory_func('Constant')('ends', ends)
        axes_X = xlf.get_xop_factory_func('Constant')('axes', axes)
        steps_X = xlf.get_xop_factory_func('Constant')('steps', steps)

        xmap = {'a': aX, 'starts': starts_X, 'ends': ends_X,
                'axes': axes_X, 'steps': steps_X}
        params = {}

        Xs = ol4.slice_op(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 2, 2, 1]

    def test_space_to_depth(self):
        a = np.zeros((1, 2, 6, 6), dtype=np.float32)

        node = onnx.helper.make_node(
            'SpaceToDepth',
            inputs=['a'],
            outputs=['y'],
            blocksize=2
        )

        wrapped_node = NodeWrapper(node)

        aX = xlf.get_xop_factory_func('Input')('a', list(a.shape),
                                               dtype='float32')

        xmap = {'a': aX}
        params = {}

        Xs = ol4.space_to_depth(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        # assert np.compress(c, a, axis=2) == (1, 2, 1, 3)

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [-1, 8, 3, 3]

    def test_where(self):
        condition = np.array([[1, 0], [1, 1]], dtype=np.bool)
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[9, 8], [7, 6]], dtype=np.float32)

        node = onnx.helper.make_node(
            'Equal',
            inputs=['condition', 'a', 'b'],
            outputs=['y']
        )

        wrapped_node = NodeWrapper(node)

        cX = xlf.get_xop_factory_func('Constant')('condition', condition)
        aX = xlf.get_xop_factory_func('Constant')('a', a)
        bX = xlf.get_xop_factory_func('Constant')('b', b)

        xmap = {'condition': cX, 'a': aX, 'b': bX}
        params = {}

        Xs = ol4.where(wrapped_node, params, xmap)

        assert len(Xs) == 1
        X = Xs[0]

        assert X.name == 'y'
        assert 'AnyOp' in X.type
        assert X.shapes.tolist() == [2, 2]
