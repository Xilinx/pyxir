# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Compile TVM model for Xilinx Vitis-AI acceleration
==================================================

This example shows how to build a TVM convolutional neural network 
model with Relay for Vitis-AI acceleration
"""

import os
import numpy as np
import logging
from pathlib import Path

import pyxir
# Import edge DPU target inside pyxir
import pyxir.contrib.target.DPUCZDX8G

import tvm
from tvm import contrib
import tvm.relay as relay
from tvm.relay import transform
from tvm.contrib import utils, graph_executor
from tvm.contrib.target import vitis_ai
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import partition_for_vitis_ai

logging.basicConfig()
logger = logging.getLogger('pyxir')
# Uncomment following line for logging
# logger.setLevel(logging.INFO)

FILE_DIR   = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(str(Path.home()),
                        'CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/')

if not os.path.exists(DATA_DIR):
    raise ValueError("Could not find directory "
                     "~/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/."
                     " Please install using following commands before"
                     " running this example: \n"
                     " $ python3 -m ck pull repo:ck-env\n"
                     " $ python3 -m ck install package:imagenet-2012-val-min")
    

###########################################################
# Define utility functions
###########################################################

def softmax(x):        
        x_exp = np.exp(x - np.max(x))
        return x_exp / x_exp.sum()

def transform_image(image):
    """"Preprocessing function"""
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


def inputs_func(img_files):
    """Utility function to read images from a list"""
    inputs = []
    for img_path in img_files:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize((224,224))
       
        inputs.append(transform_image(img))
    return inputs


######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
###############################################################################
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image

block = get_model('resnet18_v1', pretrained=True)
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
img_path = download_testdata(img_url, 'cat.png', module='data')
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

# Create test image
image = Image.open(img_path).resize((224, 224))
image = transform_image(image)

###############################################################################
# MODEL SETTINGS
#
# Parameter settings for compiling a model using tvm-vai flow
# shape_dict     : dictionary of input names as keys and input shapes as values
#                  dict{input_name:input_shape}
# target         : hardware accelerator to run the compiled model
#                    Cloud: 'DPUCADX8G'
#                    Edge: 'DPUCZDX8G-zcu104', 'DPUCZDX8G-zcu102'
###############################################################################

shape_dict = {'data': image.shape}
input_name = 'data'
out_shape = (1, 1000)
target = 'DPUCZDX8G-zcu104'

###############################################################################
# PARTITION & BUILD
# 
# Module pass to partition Relay for Vitis-AI acceleration. Targets can be 
# dpuv1, dpuv2-zcu104 and dpuv2-zcu102
# Afterwards build graph, lib and params using standard TVM flow.
##############################################################################

tvm_target = 'llvm'
lib_kwargs = {}

mod, params = relay.frontend.from_mxnet(block, shape_dict)
mod = relay.transform.InferType()(mod)
mod["main"] = bind_params_by_name(mod["main"], params)
mod = transform.RemoveUnusedFunctions()(mod)

# For the edge target we recommend converting the layout to NHWC for best performance
desired_layouts = {'nn.conv2d': ['NHWC', 'default']}
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts),
                                        relay.transform.FoldConstant()])
with tvm.transform.PassContext(opt_level=3):
     mod = seq(mod)

mod = partition_for_vitis_ai(mod, params, dpu=target)

# Convert convolutions that won't be executed on DPU back to NCHW
desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts),
                                        relay.transform.FoldConstant()])
with tvm.transform.PassContext(opt_level=3):
     mod = seq(mod)

# Build for edge target `target`
# Set Vitis AI export runtime module config to serializae the compilation information
#   to be loaded again later in this script for aarch64 & DPU cross compilation
export_rt_mod_file = os.path.join(os.getcwd(), 'vitis_ai.rtmod')
build_options = {
    'dpu': target,
    'export_runtime_module': export_rt_mod_file
}
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):   
	lib = relay.build(mod, tvm_target, params=params)


############################################################
## Create TVM InferenceSession
############################################################
print("Create InferenceSession")

InferenceSession = graph_executor.GraphModule(lib["default"](tvm.cpu()))

############################################################
## Quantization using first N inputs
## 
## Usually, to be able to accelerate inference of Neural 
## Network models with Vitis-AI DPU accelerators, those models 
## need to quantized upfront. In the ONNXRuntime Vitis-AI 
## execution provider we make use of on-the-fly quantization 
## to remove this additional preprocessing step. In this flow,
## one doesn't need to quantize his/her model upfront but can 
## make use of the typical inference execution calls 
## (InferenceSession.run) to quantize the model on-the-fly 
## using the first N inputs. This will set up and calibrate
## the Vitis-AI DPU and from that point onwards inference 
## will be accelerated for all next inputs.
############################################################
#
## Set the number of inputs used for quantization to e.g. 8 
## using the PX_QUANT_SIZE environment variable if you want
## to quantize on fewer inputs. The default is 128.
#
px_quant_size = int(os.environ['PX_QUANT_SIZE']) \
    if 'PX_QUANT_SIZE' in os.environ else 128

print("Quantize on first {} inputs".format(px_quant_size))

file_dir = DATA_DIR
img_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir)
             if f.endswith(('JPEG', 'jpg', 'png'))][:px_quant_size]
inputs = inputs_func(img_files)
print('Loaded {} inputs successfully.'.format(len(inputs)))

for i in range(px_quant_size):
    InferenceSession.set_input(input_name, inputs[i])
    InferenceSession.run()

###############
# Edge export #
###############

# !! Export TVM lib, NOTE that this also export compilation files needed later for
#   cross compilation for the edge target
lib.export_library('lib_tmp.so')

# Export lib for aarch64 target
tvm_target = tvm.target.arm_cpu('ultra96')
lib_kwargs = {
    'fcompile': contrib.cc.create_shared,
    'cc': "/usr/aarch64-linux-gnu/bin/ld"
}
build_options = {
    'load_runtime_module': export_rt_mod_file
}
with tvm.transform.PassContext(opt_level=3, config={'relay.ext.vitis_ai.options': build_options}):
    lib_edge_dpu = relay.build(mod, tvm_target, params=params)
    lib_edge_dpu.export_library('lib_dpu.so', **lib_kwargs)
