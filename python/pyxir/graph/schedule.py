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
Schedule definition


"""

import copy

from collections import OrderedDict


class Schedule(object):

    def __init__(self, time_to_layer=None, layer_to_time=None, name=None):

        self.time_to_layer = time_to_layer if time_to_layer is not None else {}
        self.layer_to_time = layer_to_time if layer_to_time is not None else {}
        self.name = name if name is not None else ""
        self.III = len(self.time_to_layer)

        assert(len(self.time_to_layer) == len(self.layer_to_time))

    def get_layer_names(self):
        # type: () -> List[str]
        """
        """
        return [self.time_to_layer[i][0] for i in range(self.III)]

    def append(self, layer_name):
        # type: (str) -> None
        if layer_name in self:
            raise ValueError("Could not append: {} because Schedule entries"
                             " should be unique".format(layer_name))

        self.time_to_layer[self.III] = [layer_name]
        self.layer_to_time[layer_name] = self.III
        self.III += 1

    def __contains__(self, layer_name):
        # type: (str) -> boolean
        return layer_name in self.layer_to_time

    def __len__(self):
        # type: () -> int
        return self.III

    def __dict__(self):
        # type: () -> dict
        return OrderedDict([
            ('time_to_layer', self.time_to_layer),
            ('layer_to_time', self.layer_to_time),
            ('name', self.name)
        ])

    def __deepcopy__(self, memo):
        # type: (dict) -> Schedule
        """
        """
        copy_time_to_layer = copy.deepcopy(self.time_to_layer)
        copy_layer_to_time = copy.deepcopy(self.layer_to_time)

        return Schedule(
            time_to_layer=copy_time_to_layer,
            layer_to_time=copy_layer_to_time,
            name=self.name
        )
