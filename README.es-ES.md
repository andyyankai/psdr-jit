

[![PyPI version](https://badge.fury.io/py/psdr-jit.svg?kill_cache=1)](https://badge.fury.io/py/psdr-jit)
[![DOI](https://zenodo.org/badge/530067467.svg)](https://zenodo.org/doi/10.5281/zenodo.13777324)
# psdr-jit
Renderizador diferenciable en espacio de trayectorias con [`drjit`](https://drjit.readthedocs.io/en/latest/) como backend numérico.
Repositorio antiguo: https://github.com/uci-rendering/psdr-cuda


## Instalación
Utilizamos [`OptiX 7`](https://developer.nvidia.com/rtx/ray-tracing/optix) para acelerar el trazado de rayos. Asegúrate de tener instalados [`cudatoolkit`](https://developer.nvidia.com/cuda-toolkit) (puedes instalarlo usando `conda`) y un controlador de pantalla NVIDIA compatible con `OptiX 7`. La versión más reciente que probamos fue `cudatoolkit==11.7`. Luego, ejecuta
```bash
pip install psdr-jit
```

## Instalación local
Para instalar `psdr-jit` localmente, primero clona el repositorio de forma recursiva (para incluir los submódulos de git):
```bash
git clone --recursive https://github.com/andyyankai/psdr-jit.git
```
o bien
```bash
git clone https://github.com/andyyankai/psdr-jit.git
cd psdr-jit
git submodule update --init --recursive
```
Luego, compila e instala `psdr-jit` mediante
```bash
pip install .
```
o bien
```bash
pip install --no-build-isolation -ve .
```

Ten en cuenta que también instalará `drjit`, ya que `psdr-jit` depende de él. También se requiere `cudatoolkit`.

## Primeros pasos
```python
import drjit
import psdr_jit
```

### Ejemplo de renderizado diferencial

```python
import cv2
import sys
import torch

import psdr_jit as psdr
import drjit
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD
from drjit.cuda import Float as FloatC, Matrix4f as Matrix4fC

sc = psdr.Scene()
sc.opts.spp = 32 # Interior Term
sc.opts.sppe = 32 # Primary Edge
sc.opts.sppse = 32 # Secondary Edge
sc.opts.height = 512
sc.opts.width = 512

integrator = psdr.PathTracer(3)	


sensor = psdr.PerspectiveCamera(60, 0.000001, 10000000.)
to_world = Matrix4fD([[1.,0.,0.,208.],
                     [0.,1.,0.,273.],
                     [0.,0.,1.,-800.],
                     [0.,0.,0.,1.],])
sensor.to_world = to_world
sc.add_Sensor(sensor)

sc.add_BSDF(psdr.DiffuseBSDF([0.0, 0.0, 0.0]), "light")
sc.add_BSDF(psdr.DiffuseBSDF(), "cat")
sc.add_BSDF(psdr.DiffuseBSDF([0.95, 0.95, 0.95]), "white")
sc.add_BSDF(psdr.DiffuseBSDF([0.20, 0.90, 0.20]), "green")
sc.add_BSDF(psdr.DiffuseBSDF([0.90, 0.20, 0.20]), "red")

sc.add_Mesh("./data/objects/cbox/cbox_luminaire.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,-0.5],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "light", psdr.AreaLight([20.0, 20.0, 8.0]))
sc.add_Mesh("./data/objects/cbox/cbox_smallbox.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "cat", None)
sc.add_Mesh("./data/objects/cbox/cbox_largebox.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "cat", None)
sc.add_Mesh("./data/objects/cbox/cbox_floor.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
sc.add_Mesh("./data/objects/cbox/cbox_ceiling.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
sc.add_Mesh("./data/objects/cbox/cbox_back.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
sc.add_Mesh("./data/objects/cbox/cbox_greenwall.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "green", None)
sc.add_Mesh("./data/objects/cbox/cbox_redwall.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "red", None)

P = FloatD(0.)
drjit.enable_grad(P)

sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,P*100.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],]))


sc.configure()
sc.configure([0])

img = integrator.renderD(sc, 0)
org_img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("psdr_jit_forward.exr", output)


drjit.set_grad(P, 1.0)
drjit.forward_to(img)
diff_img = drjit.grad(img)
diff_img = diff_img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("psdr_jit_diff_debug.exr", output)
```

### Más tutoriales
Tutorial de evaluación de gradientes en la carpeta [`tutorial`](https://github.com/andyyankai/psdr-jit/tree/main/tutorials)

Ejemplo de optimización a nivel de objeto: [`Neural-PBIR`](https://neural-pbir.github.io/) (ICCV 2023)

Ejemplo de optimización a nivel de escena: [`psdr-room`](https://github.com/andyyankai/psdr-room) (SIGGRAPH ASIA 2023)

Ejemplo de optimización SDF+Nerf: por publicar

### Proyectos basados en psdr-jit
Por definir

Punto de control libre de IA: 476ba6ef39e24474e85953ba3743bb0b967c797e
