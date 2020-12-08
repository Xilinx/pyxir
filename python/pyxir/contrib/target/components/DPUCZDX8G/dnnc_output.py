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

""" Module wrapping parsing DNNC compiler output """

import abc


class BaseDNNCOutput(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, output):
        self.d = self._parse(output)

    @abc.abstractmethod
    def _parse(self):
        raise NotImplementedError("")

    @abc.abstractmethod
    def get_input_nodes(self):
        raise NotImplementedError("")

    @abc.abstractmethod
    def get_output_nodes(self):
        raise NotImplementedError("")


class DNNCOutput(BaseDNNCOutput):

    def _parse(self, output):
        # type: (str) -> dict
        """ Parse DNNC v4.0 compiler output """

        d = {
            'Boundary Input Tensors': {},
            'Boundary Output Tensors': {},
            'Boundary Output Tensors Shapes': {},
            'Input Nodes': {},
            'Output Nodes': {}
        }
        lines = output.split('\n')[0].split('\\n')

        for idx, line in enumerate(lines):
            split_line = line.lstrip().rstrip().split(" : ")

            if split_line[0] == 'Kernel ID':
                d['Kernel ID'] = lines[idx + 1].split(" : ")[0]
                d['Name'] = lines[idx + 1].split(" : ")[1]
            elif split_line[0] == 'Kernel Name':
                d['Kernel Name'] = split_line[1]
            elif split_line[0] == 'Kernel Type':
                d['Kernel Type'] = split_line[1]
            elif split_line[0] == 'Code Size':
                d['Code Size'] = split_line[1]
            elif split_line[0] == 'Param Size':
                d['Param Size'] = split_line[1]
            elif split_line[0] == 'Workload MACs':
                d['Workload MACs'] = split_line[1]
            elif split_line[0] == 'IO Memory Space':
                d['IO Memory Space'] = split_line[1]
            elif split_line[0] == 'Mean Value':
                d['Mean Value'] = split_line[1].split(',')
            elif split_line[0] == 'Node Count':
                d['Node Count'] = split_line[1]
            elif split_line[0] == 'Tensor Count':
                d['Tensor Count'] = split_line[1]
            elif split_line[0] == 'Total Tensor Count':
                d['Tensor Count'] = split_line[1]
            elif split_line[0] == 'Boundary Input Tensor(s)   (H*W*C)':
                for i in range(idx + 1, len(lines)):
                    split_line_i = lines[i].lstrip().rstrip().split(" : ")

                    if len(split_line_i) != 2:
                        break

                    name, shape = split_line_i

                    if shape in d['Boundary Input Tensors']:
                        raise ValueError("DNNC compiler cannot handle multiple"
                                         " inputs with the same shape")

                    d['Boundary Input Tensors'][shape] = name.split(":")[0]
            elif split_line[0] == 'Boundary Output Tensor(s)   (H*W*C)':
                for i in range(idx + 1, len(lines)):
                    split_line_i = lines[i].lstrip().rstrip().split(" : ")

                    if len(split_line_i) != 2:
                        break

                    name, shape = split_line_i
                    name = name.split(":")[0]

                    # if shape in d['Boundary Output Tensors']:
                    #     raise ValueError("DNNC compiler cannot handle multiple"
                    #                      " outputs with the same shape")
                    
                    d['Boundary Output Tensors'][name] = name + ':0'
                    d['Boundary Output Tensors Shapes'][shape] = name + ':0'
            elif split_line[0] == 'Total Node Count':
                d['Total Node Count'] = split_line[1]
            elif split_line[0] in ['Input Node(s)   (H*W*C)',
                                   'Input Node(s)(H*W*C)']:
                for i in range(idx + 1, len(lines)):
                    split_line_i = lines[i].lstrip().rstrip().split(" : ")

                    if len(split_line_i) != 2:
                        break

                    name, shape = split_line_i

                    if shape in d['Input Nodes']:
                        raise ValueError("DNNC compiler cannot handle multiple"
                                         " inputs with the same shape")

                    d['Input Nodes'][shape] = name[:-3]

            elif split_line[0] in ['Output Node(s)   (H*W*C)',
                                   'Output Node(s)(H*W*C)']:
                for i in range(idx + 1, len(lines)):
                    split_line_i = lines[i].lstrip().rstrip().split(" : ")

                    if len(split_line_i) != 2:
                        break

                    name, shape = split_line_i

                    # if shape in d['Output Nodes']:
                    #     raise ValueError("DNNC compiler cannot handle multiple"
                    #                      " outputs with the same shape")

                    d['Output Nodes'][name[:-3]] = name[:-3]
                    d['Output Nodes'][shape] = name[:-3]

        return d

    def get_input_nodes(self):
        # type: () -> Dict[str, str]
        return self.d['Input Nodes']

    def get_output_nodes(self):
        # type: () -> Dict[str, str]
        return self.d['Boundary Output Tensors']

    def get_output_nodes_on_shapes(self):
        # type: () -> Dict[str, str]
        return self.d['Boundary Output Tensors Shapes']

    def get_dnnc_str(self, str_value: str) -> str:
        return str_value.replace('-', '_')
