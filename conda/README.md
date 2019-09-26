# Build PyXIR conda environment for ONNX flow

```
conda env create -f environment.yml

conda activate pyxir-onnx

export PYTHONPATH={PATH-TO-PYXIR}/python:${PYTHONPATH}

cd {PATH-TO-PYXIR}/python

python3 setup.py build_ext --inplace --force
```

## Test

```
python3

>>> import pyxir
```