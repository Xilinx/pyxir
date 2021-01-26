# TVM Examples

## Setup

For seting up the host and edge environment, please refer to the [TVM Vitis AI documentation](https://tvm.apache.org/docs/deploy/vitis_ai.html).

## Example files

This folder contains two scripts for compiling and running a ResNet 18 example on the ZCU104 board with Vitis AI DPU acceleration.
The `edge_resnet_18_host.py` script should be executed on a host x86 machine and will download, set up and compile a ResNet 18 model
for DPU execution. This will save a lib_dpu.so file containing the aarch64 cross compiled TVM - DPU inference module to the current
work directory. This file should be moved to the ZCU104 together with the `edge_resnet_18_board.py` script and can be executed with

```
python3 edge_resnet_18_board.py lib_dpu.so
```
