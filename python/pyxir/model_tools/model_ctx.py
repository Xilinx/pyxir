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
Module for ModelCtx definition and implementation


"""

import os


class ModelCtx(object):

    """
    Data structure for keeping track of the models context we need for
    compilation, quantization, inference and validation
    """

    def __init__(self,
                 model_name,
                 framework,
                 model_path,
                 opt_model_path,
                 inputs,
                 input_layouts,
                 input_shapes,
                 preprocessing,
                 input_dir,
                 outputs,
                 postprocessing,
                 scoring):

        self.model_name = model_name
        self.framework = framework
        self.model_path = model_path
        self.opt_model_path = opt_model_path

        self.inputs = inputs
        self.input_layouts = input_layouts
        self.input_shapes = input_shapes
        self.preprocessing = preprocessing
        self.input_dir = input_dir

        self.outputs = outputs if outputs not in ["None", "none"] else None
        self.postprocessing = postprocessing
        self.scoring = scoring

    def get_model_name(self):
        return self.model_name

    def get_framework(self):
        return self.framework

    def get_model_path(self):
        return self.model_path

    def get_opt_model_path(self):
        return self.opt_model_path

    def get_inputs(self):
        return self.inputs

    def get_input_layouts(self):
        return {i: f for i, f in zip(self.inputs, self.input_layouts)}

    def get_input_shapes(self):
        return {i: s for i, s in zip(self.inputs, self.input_shapes)}

    def get_preprocessing(self):
        return {i: p for i, p in zip(self.inputs, self.preprocessing)}

    def get_input_dir(self):
        return self.input_dir

    def get_test_inputs(self):
        # type: () -> List[str]
        """ Returns test inputs as a list of input paths """
        return [os.path.join(self.input_dir['test'], file_name)
                for file_name in os.listdir(self.input_dir['test'])
                if (file_name.endswith('.jpg') or
                    file_name.endswith('.png') or file_name.endswith('.JPEG'))]

    def get_val_inputs(self):
        # type: () -> List[str]
        """ Returns validation inputs as a list of input paths """
        return [os.path.join(self.input_dir['validation'], file_name)
                for file_name in os.listdir(self.input_dir['validation'])
                if (file_name.endswith('.jpg') or
                    file_name.endswith('.png') or file_name.endswith('.JPEG'))]

    def get_outputs(self):
        return self.outputs

    def get_postprocessing(self):
        return self.postprocessing

    def get_scoring(self):
        return self.scoring
