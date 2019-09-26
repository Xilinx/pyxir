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

""" Data structure for quantizer output """


class QuantizerOutputElem(object):

    def __init__(self, q_key, q_file, q_info, orig_pb):
        self.q_key = q_key
        self.q_file = q_file
        self.q_info = q_info
        self.orig_pb = orig_pb

    def set_q_file(self, q_file):
        self.q_file = q_file

    def get_q_file(self):
        return self.q_file

    def set_orig_pb(self, orig_pb):
        self.orig_pb = orig_pb

    def get_orig_pb(self):
        return self.orig_pb

    def set_q_info(self, q_info):
        self.q_info = q_info

    def get_q_info(self):
        return self.q_info


class QuantizerOutput(object):

    def __init__(self, name):
        self.name = name
        self.data = {}

    def get_name(self):
        return self.name

    def keys(self):
        return self.data.keys()

    def add(self, q_key, q_file, q_info, orig_pb):
        self.data[q_key] = QuantizerOutputElem(q_key, q_file, q_info, orig_pb)

    def get_q_file(self, q_key):
        return self.data[q_key].get_q_file()

    def get_q_info(self, q_key):
        return self.data[q_key].get_q_info()

    def get_orig_pb(self, q_key):
        return self.data[q_key].get_orig_pb()
