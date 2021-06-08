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
Run a cross compiled TVM - DPU module on the board
==================================================
"""

import os
import sys
import time
import logging
from pathlib import Path
from PIL import Image

import numpy as np
import pyxir

logger = logging.getLogger('pyxir')
# Uncomment following line for logging
# logger.setLevel(logging.INFO)

import tvm
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


###########################################################
# Define utility functions
###########################################################

def softmax(x):        
    x_exp = np.exp(x - np.max(x))
    return x_exp / x_exp.sum()

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


############################################################
## Download synset
############################################################
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')

synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])

synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

out_shape =(1,1000)


############################################################
## Load TVM - DPU runtime module
############################################################

lib_path  = os.path.join(FILE_DIR, sys.argv[1])
lib = tvm.runtime.module.load_module(lib_path)
mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))


###########################################################
# Accelerated inference on new image
###########################################################

image = Image.open(img_path).resize((224, 224))
image = transform_image(image)
map_inputs= {"data": image}


for name, data in map_inputs.items():
    mod.set_input(name, data)
start = time.time()
mod.run()
end = time.time()


###########################################################
# Postprocessing
###########################################################

out_shape = (1, 1000)
out = tvm.nd.empty(out_shape)
res = softmax(mod.get_output(0, out).asnumpy()[0])
top1 = np.argmax(res)

print('========================================')
print('TVM prediction top-1:', top1, synset[top1])
print('========================================')

inference_time = np.round((end - start) * 1000, 2)
print('========================================')
print('Inference time: ' + str(inference_time) + " ms")
print('========================================')
