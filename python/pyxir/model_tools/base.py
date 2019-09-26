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
Module for loading model paths from different frameworks and returning them
as a dictionary


"""

import os
import json

from .model_ctx import ModelCtx

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_models_dict(models_dir=os.path.join(FILE_DIR, '../../../lib/models'),
                    check_exists=True):
    # (str, bool) -> dict

    d = {}

    models_dir = models_dir if os.path.isabs(models_dir) \
        else os.path.join(os.getcwd(), models_dir)

    with open(os.path.join(models_dir, 'models.json')) as f:
        models = json.load(f)

    found_models = {}
    for model_name in models:
        abs_model_path = \
            os.path.join(models_dir, models[model_name]['model_path'])
        abs_opt_model_path = \
            os.path.join(models_dir, models[model_name]['opt_model_path'])

        abs_input_dir_base = \
            os.path.join(models_dir, models[model_name]['input_dir']['base'])
        abs_input_dir_test = \
            os.path.join(models_dir, models[model_name]['input_dir']['test'])
        abs_input_dir_val = \
            os.path.join(models_dir,
                         models[model_name]['input_dir']['validation'])

        abs_scoring_synset = \
            os.path.join(models_dir,
                         models[model_name]['scoring']['synset']) if\
            models[model_name]['scoring']['synset'] != "None" else "None"
        abs_scoring_val = \
            os.path.join(models_dir,
                         models[model_name]['scoring']['validation']) if\
            models[model_name]['scoring']['validation'] != "None" else "None"

        if check_exists and not os.path.exists(abs_model_path):
            print("[WARNING]: model: {} could not be found. See models"
                  " directory for setup".format(model_name))
        elif not os.path.exists(abs_input_dir_base):
            print("[WARNING]: model: {} base input directory: {} could not be"
                  " found. See models directory for setup"
                  .format(model_name, abs_input_dir_base))
        elif not os.path.exists(abs_input_dir_test):
            print("[WARNING]: model: {} test input directory: {} could not be"
                  " found. See models directory for setup"
                  .format(model_name, abs_input_dir_test))
        elif not os.path.exists(abs_input_dir_val):
            print("[WARNING]: model: {} validation input directory: {} could"
                  " not be found. See models directory for setup"
                  .format(model_name, abs_input_dir_val))
        elif abs_scoring_synset != "None" and \
                not os.path.exists(abs_scoring_synset):
            print("[WARNING]: model: {} synset file: {} could"
                  " not be found. See models directory for setup"
                  .format(model_name, abs_input_dir_val))
        elif abs_scoring_val != "None" and \
                not os.path.exists(abs_scoring_val):
            print("[WARNING]: model: {} validation score file: {} could"
                  " not be found. See models directory for setup"
                  .format(model_name, abs_input_dir_val))
        else:
            if os.path.exists(abs_opt_model_path):
                models[model_name]['opt_model_path'] = abs_opt_model_path

            models[model_name]['model_path'] = abs_model_path
            models[model_name]['input_dir']['base'] = abs_input_dir_base
            models[model_name]['input_dir']['test'] = abs_input_dir_test
            models[model_name]['input_dir']['validation'] = abs_input_dir_val

            models[model_name]['scoring']['synset'] = abs_scoring_synset
            models[model_name]['scoring']['validation'] = abs_scoring_val

            found_models[model_name] = models[model_name]

    return found_models


def get_model_context(model_dict):
    # type: (dict) -> ModelCtx
    """ Returns the model context from a dictionary """
    return ModelCtx(
        model_name=model_dict['model'],
        framework=model_dict['framework'],
        model_path=model_dict['model_path'],
        opt_model_path=model_dict['opt_model_path'],
        inputs=model_dict['inputs'],
        input_layouts=model_dict['input_layouts'],
        input_shapes=model_dict['input_shapes'],
        preprocessing=model_dict['preprocessing'],
        input_dir=model_dict['input_dir'],
        outputs=model_dict['outputs'],
        postprocessing=model_dict['postprocessing'],
        scoring=model_dict['scoring']
    )
