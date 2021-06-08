# PyXIR

PyXIR is an Neural Network Intermediate Representation (IR) for deep learning. It is designed to be an interface between deep learning frameworks and neural network hardware accelerators, specifically Xilinx Vitis-AI FPGA based accelerators like [DPU](https://www.xilinx.com/products/intellectual-property/dpu.html). 

At the moment PyXIR integrates with following frameworks:
* [ONNXRuntime](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/Vitis-AI-ExecutionProvider.md)
* [TVM Open Deep Learning compiler stack](https://tvm.apache.org/docs/deploy/vitis_ai.html)

and with following Vitis-AI accelerators:
* DPUCADX8G (formerly DPUv1)
* DPUCZDX8G (formerly DPUv2)

Note that not all accelerators are enabled through all frameworks at the moment. For example, through the ONNXRuntime framework only the DPUCADX8G accelerator is supported for now.
