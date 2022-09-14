# psdr-jit
Path-space differentiable renderer

Official documentation: https://psdr-jit.readthedocs.io/en/latest/

To install project, you simply need to
```bash
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 16 2019" -A x64 -D PYTHON_ROOT=C:/ProgramData/Anaconda3 -D OptiX_INSTALL_DIR=C:\ubuntu\psdr_jit\ext_win64\optix
cmake --build build --target install --config Release
```

## Getting Started
```bash
import drjit
import psdr_jit
```