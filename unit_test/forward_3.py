import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path
import torch
import numpy as np
import psdr_jit as psdr
import drjit
# drjit.set_flag(drjit.JitFlag.PrintIR, True)
from drjit.cuda.ad import Array3f as Vector3fD
from drjit.cuda.ad import Float as FloatD, Matrix3f as Matrix3fD
from drjit.cuda import Array3f as Vector3fC


def translateT(t_vec):
    device = t_vec.device
    dtype = t_vec.dtype 

    to_world = torch.eye(3).to(device).to(dtype)
    to_world[:2, 2] = t_vec

    return to_world

def rotateT(angle, use_degree=True):
    device = angle.device
    dtype = angle.dtype 

    to_world = torch.eye(3).to(device).to(dtype)
    if not torch.is_tensor(angle):
        angle = torch.tensor(angle).to(device).to(dtype)
    if use_degree:
        angle = torch.deg2rad(angle)

    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    R = cos_theta * torch.eye(2).to(device).to(dtype)

    R[0 ,1] = -sin_theta
    R[1 ,0] = sin_theta

    to_world[:2, :2] = R

    return to_world

def scaleT(size):
    device = size.device
    dtype = size.dtype 

    to_world = torch.eye(3).to(device).to(dtype)

    if size.size(dim=0) == 1:
        to_world[:2, :2] = torch.diag(size).to(device).to(dtype) * torch.eye(2).to(device).to(dtype)
    elif size.size(dim=0) == 2:
        to_world[:2, :2] = torch.diag(size).to(device).to(dtype)
    else:
        print("error transform.py for scale")
        exit()

    return to_world



output_path = Path('result','inv_3')
output_path.mkdir(parents=True, exist_ok=True)

sc = psdr.Scene()
sc.load_file("texture.xml")
sc.opts.spp = 32
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.configure()
integrator = psdr.CollocatedIntegrator(200)	

P = FloatD(0.)

drjit.enable_grad(P)


sc.param_map["BSDF[0]"].reflectance.to_world = Matrix3fD([[1.,0.,0.],
														 [0.,1.,P],
														 [0.,0.,1.],])
sc.configure()
img = integrator.renderD(sc, 0)

drjit.set_grad(P, 1.0)
drjit.forward_to(img)
diff_img = drjit.grad(img)
diff_img = diff_img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)

cv2.imwrite("texture_to_world.exr", output)
eps = 0.001
sc.param_map["BSDF[0]"].reflectance.to_world = Matrix3fD([[1.,0.,0.],
                                                         [0.,1.,-eps],
                                                         [0.,0.,1.],])
sc.configure()
fd1_img = integrator.renderC(sc, 0)

sc.param_map["BSDF[0]"].reflectance.to_world = Matrix3fD([[1.,0.,0.],
                                                         [0.,1.,eps],
                                                         [0.,0.,1.],])
sc.configure()
fd2_img = integrator.renderC(sc, 0)

diff_img = (fd2_img-fd1_img) / (2.0*eps)
diff_img = diff_img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
cv2.imwrite("FD_texture_to_world.exr", output)

