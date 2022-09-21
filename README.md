# psdr-jit
Path-space differentiable renderer

Official documentation: https://psdr-jit.readthedocs.io/en/latest/

To install project, you simply need to

### On Windows:
```bash
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 16 2019" -A x64 -D PYTHON_ROOT=C:/ProgramData/Anaconda3 -D OptiX_INSTALL_DIR=C:\ubuntu\psdr_jit\ext_win64\optix
cmake --build build --target install --config Release
```
then add /build/python/ to your PYTHONPATH

### On Ubuntu:
```bash
git submodule update --init --recursive
cmake -S . -B build -D PYTHON_ROOT=/usr/include/python3.8/ -D OptiX_INSTALL_DIR=../optix -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-
cmake --build build --target install --config Release -j
```
then add /build/python/ to your PYTHONPATH
add /build/python/drjit to your LD_LIBRARY_PATH
add /build/python/psdr_jit to your LD_LIBRARY_PATH

## Getting Started
```bash
import drjit
import psdr_jit
```