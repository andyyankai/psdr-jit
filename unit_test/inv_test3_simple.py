from ast import Delete
import os
import time
from xmlrpc.client import Boolean
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path
import torch
import numpy as np
import random

test_psdrjit = int(sys.argv[1])
rnd_camera = True

if test_psdrjit:
    import psdr_jit as psdr
    import drjit
    # drjit.set_flag(drjit.JitFlag.PrintIR, True)
    from drjit.cuda.ad import Array3f as Vector3fD
    from drjit.cuda import Array3f as Vector3fC
    from drjit import dot as dot
    from drjit import backward as backward
    from drjit import grad as grad

else:
    import psdr_cuda as psdr
    import enoki
    from enoki import *
    from enoki.cuda_autodiff import Vector3f as Vector3fD
    from enoki.cuda import Vector3f as Vector3fC
    from enoki import dot as dot
    from enoki import backward as backward
    from enoki import gradient as grad

output_path = Path('test_inv3_simple')
output_path.mkdir(parents=True, exist_ok=True)
integrator = psdr.PathTracer(1)

# load scene
old_path = os.getcwd()
tmp_path = Path("data/butterfly_flower")
sc = psdr.Scene()
os.chdir(tmp_path)
sc.load_file("scene.xml", False)
os.chdir(old_path)
num_pixels = sc.opts.width * sc.opts.height
t_res = 1024
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0
sc.opts.spp = 8

tar_images = []
target_images_path = Path(r'.\data\butterfly_flower\tar_images')
for target_image_path in sorted(target_images_path.glob('*.exr'), key=lambda x : int(x.stem)):
    img = torch.from_numpy(cv2.imread(str(target_image_path), -1))
    tar_images.append(img.to('cuda').to(torch.float32))

# Optimization
num_sensors = sc.num_sensors
if rnd_camera:
    sensor_ids = list(range(num_sensors))
    random.shuffle(sensor_ids)
else:
    sensor_ids = [0] * num_sensors
iter = 0
for sensor_id in sensor_ids:
    t0 = time.process_time()
    # sc.configure2([sensor_id])
    sc.configure()
    image = integrator.renderC(sc, sensor_id).numpy().reshape((sc.opts.height, sc.opts.width, 3))
    t1 = time.process_time()
    print(f"[Iter {iter}] sensor_id: {sensor_id} | time: {t1 - t0}")
    iter += 1
    image = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)).to('cuda')
    image_vis = torch.cat([image, tar_images[sensor_id]],
                           dim=1)
    cv2.imwrite(str(output_path) + f"/iter_{iter}.exr", image_vis.to('cpu').numpy())
