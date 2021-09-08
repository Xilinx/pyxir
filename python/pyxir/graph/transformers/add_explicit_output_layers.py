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
"""Module for the AddExplicitOutputLayers pass"""

import numpy as np
import pyxir as px

from typing import List

from .. import XGraph
from ..passing import XGraphMutator
from ..layer.xlayer import XLayer


class AddExplicitOutputLayers(XGraphMutator):
    """Add explicit output layers"""

    def __init__(self, out_tensor_names: List[str] = None, layout: str = None):
        super().__init__()
        self.out_tensor_names = (
            set(out_tensor_names) if out_tensor_names is not None else set([])
        )
        self.out_tensor_map = {}
        self.layout = layout

    def transform(self, xgraph: XGraph) -> XGraph:
        """Add XGraph output names to out tensor names attribute"""
        self.out_tensor_names |= set(xgraph.get_output_names())
        return xgraph

    def visit(self, X: XLayer) -> XLayer:
        if (
            X.name in self.out_tensor_names
            and len(X.tops) > 0
            and len(X.shapes) == 4
            and self.layout is not None
        ):
            layer_name = X.name
            new_name = X.name + "_hidden"
            X.name = new_name
            self.out_tensor_map[layer_name] = new_name

            if any([b in self.out_tensor_map for b in X.bottoms]):
                X.bottoms[:] = [
                    b if b not in self.out_tensor_map else self.out_tensor_map[b]
                    for b in X.bottoms
                ]

            channels = X.shapes[self.layout.index("C")]
            weights = np.identity(channels, dtype=np.float32).reshape(
                channels, channels, 1, 1
            )
            wX = px.ops.constant(new_name + "_w", weights)
            idX = px.ops.conv2d(
                layer_name, X, wX, kernel_size=[1, 1], data_layout=self.layout
            )

            return [X, idX]
        elif any([b in self.out_tensor_map for b in X.bottoms]):
            X.bottoms[:] = [
                b if b not in self.out_tensor_map else self.out_tensor_map[b]
                for b in X.bottoms
            ]
            return X

        return super().visit(X)
