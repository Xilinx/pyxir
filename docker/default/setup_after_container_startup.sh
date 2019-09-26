#!/usr/bin/env bash

#export TVM_HOME=$MLSUITE_ROOT/3rdparty/tvm
#export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:$TVM_HOME/vta/python:${PYTHONPATH}

export PYXIR_ROOT="/opt/pyxir/python"
export PYTHONPATH=$PYXIR_ROOT:${PYTHONPATH}

bash
