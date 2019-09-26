# Build Vitis-AI docker environment

Build and run the Vitis-AI - pyxir - TVM docker environment

```
cd [PYXIR_ROOT]/docker/vai

./docker_build.sh

cd [WORKSPACE]
.[PATH_TO_PYXIR]/docker/vai/docker_run.sh

$ export PYXIR_HOME=[PATH_TO_PYXIR]/python
```

Quick test docker environment

```
conda activate vitis-ai-tensorflow

vai_c_tensorflow

python3
>>> import pyxir
>>> import tensorflow
>>> from tensorflow.contrib import decent_q
```



