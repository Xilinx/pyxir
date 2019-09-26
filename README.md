# PyXIR

PyXIR is an Neural Network Intermediate Representation (IR) for deep learning. It is designed to be an interface between deep learning frameworks and neural network hardware accelerators, specifically Xilinx Vitis-AI FPGA based accelerators like [DPU](https://www.xilinx.com/products/intellectual-property/dpu.html). 

At the moment PyXIR integrates with following frameworks:
* [ONNXRuntime](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Vitis-AI-ExecutionProvider.md)
* [TVM Open Deep Learning compiler stack](https://github.com/apache/incubator-tvm) (coming up)

and with following Vitis-AI accelerators:
* DPUv1
* DPUv2

Note that not all accelerators are enabled through all frameworks at the moment. For example, through the ONNXRuntime framework only the DPUv1 accelerator is supported for now.