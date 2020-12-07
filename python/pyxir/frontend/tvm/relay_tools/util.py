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

"""Utility module for Relay to XGraph conversion"""

from typing import Any, Dict, List

class Schedule(object):
    """"Schedule for Relay to XGraph converter to keep track of operations
        in topological order"""

    def __init__(self, netmap: dict):
        self.netmap = netmap
        self.schedule = []

    def append(self, value: Any) -> None:
        self.schedule.append(value)

    def __contains__(self, value):
        return value in self.schedule

    def __iter__(self) -> Any:
        for e in self.schedule:
            yield e

    def __delete__(self, instance: Any) -> None:
        del self.schedule[instance]

    def __len__(self) -> int:
        return len(self.schedule)


def broadcast_shapes(lshape_orig: List[int], rshape_orig: List[int]):
    """Utility function for broadcasting two shapes"""
    if len(lshape_orig) >= len(rshape_orig):
        lshape = lshape_orig[:]
        rshape = [None] * (len(lshape_orig) - len(rshape_orig)) + rshape_orig[:]
    else:
        rshape = rshape_orig[:]
        lshape = [None] * (len(rshape_orig) - len(lshape_orig)) + lshape_orig[:]

    assert len(lshape) == len(rshape)

    reversed_shape = []
    for ls, rs in zip(reversed(lshape), reversed(rshape)):
        if ls == rs or ls in [1, None] or rs in [1, None]:
            if ls is None:
                reversed_shape.append(rs)
            elif rs is None:
                reversed_shape.append(ls)
            else:
                reversed_shape.append(max(ls, rs))
        else:
            raise ValueError("Invalid shapes for broadcasted additions:"
                             " {} and {}".format(lshape_orig, rshape_orig))

    return list(reversed(reversed_shape))
