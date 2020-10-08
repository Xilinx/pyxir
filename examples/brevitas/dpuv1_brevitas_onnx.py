import os

import numpy as np
import onnx
import torch
import torchvision.models as models
from brevitas.graph.quantizer import quantize, BatchNormHandling
from brevitas.inject.defaults import *
from brevitas import config
from brevitas.onnx import export_dpuv1_onnx

import pyxir
from pyxir.frontend.onnx.base import from_onnx
from pyxir.contrib.target import DPUCADX8G_external_quantizer
import pathlib

config.IGNORE_MISSING_KEYS = True
file_dir = pathlib.Path(__file__).parent.absolute()
target = 'DPUCADX8G'


IN_SIZE = (1, 3, 224, 224)


# Define quantization scheme
bias_quant = IntQuant & StatsMaxScaling & PerTensorPoTScaling8bit
weight_quant = NarrowIntQuant & StatsMaxScaling & PerTensorPoTScaling8bit
io_quant = IntQuant & ParamFromRuntimePercentileScaling & PerTensorPoTScaling8bit

# Import float model
model = models.resnet18()

# Apply quantization
inp = torch.randn(IN_SIZE)
model = quantize(
    model,
    inp,
    weight_quant=weight_quant,
    input_quant=io_quant,
    output_quant=io_quant,
    bias_quant=bias_quant,
    bn_handling=BatchNormHandling.MERGE_AND_QUANTIZE)

# Finetune the model
# . . .

# Export to ONNX
onnx_filename = 'dpuv1_resnet18.onnx'
export_dpuv1_onnx(model, input_shape=IN_SIZE, input_t=inp, export_path=onnx_filename)

# Load ONNX into PyXIR
onnx_model = onnx.load(onnx_filename)
xgraph = from_onnx(onnx_model)
xgraph = pyxir.partition(xgraph, [target])
xgraph = pyxir.optimize(xgraph, target)
work_dir = os.path.join(file_dir, f'{target}_quant_trained_resnet18_workdir')
inputs = np.random.randn(*IN_SIZE)
def inputs_func(iter): return {'inp.1': inputs}
xgraph = pyxir.quantize(xgraph, target, inputs_func, work_dir=work_dir)
pyxir.build(xgraph, target, work_dir=work_dir, build_dir=work_dir, runtime='cpu-np')