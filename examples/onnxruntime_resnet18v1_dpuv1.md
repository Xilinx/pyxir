# ONNXRuntime with Xilinx Vitis-AI acceleration example

Microsoft ONNXRuntime is a framework designed for running ONNX models on a variety of platforms.

ONNXRuntime is enabled with Vitis-AI and available through the Microsoft github page:

https://github.com/microsoft/onnxruntime

Vitis-AI documentation for ONNXRuntime is available here:

https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Vitis-AI-ExecutionProvider.md

## Setup

1. Follow setup instructions here to setup the ONNXRuntime - Vitis-AI environment: https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Vitis-AI-ExecutionProvider.md
2. Download minimal ImageNet validation dataset (step specific to this example):
   ```
   python3 -m ck pull repo:ck-env
   python3 -m ck install package:imagenet-2012-val-min
   ```
3. Install ONNX and Pillow packages
   ```
   pip3 install --user onnx pillow
   ```
4. (Optional) set the number of inputs to be used for on-the-fly quantization to a lower number (e.g. 8) to decrease the quantization time (potentially at the cost of a lower accuracy):
   ```
   export PX_QUANT_SIZE=8
   ```