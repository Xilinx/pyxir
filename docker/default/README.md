# Build default (CPU only) docker environment

Build and run the Vitis-AI - pyxir - TVM docker environment

```
cd [PYXIR_ROOT]/docker/default

./docker_build.sh

./docker_run.sh

cd python
python3 setup.py build_ext --inplace --force
cd ..
export TVM_HOME=/opt/pyxir/lib/tvm
export PYXIR_HOME=/opt/pyxir/python
export PYTHONPATH=$PYXIR_HOME:$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:$TVM_HOME/vta/python:${PYTHONPATH}
```

Quick test docker environment

```

python3
>>> import pyxir
>>> import tvm
```

