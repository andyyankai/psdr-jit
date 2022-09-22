from ast import Delete
import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path
import torch
import numpy as np
from ivt.io import write_exr, read_exr

test_psdrjit = int(sys.argv[1])

if test_psdrjit:
    import psdr_jit as psdr
    import drjit
    # drjit.set_flag(drjit.JitFlag.PrintIR, True)
    from drjit.cuda.ad import Array3f as Vector3fD
    from drjit.cuda import Array3f as Vector3fC

else:
    import psdr_cuda as psdr
    import enoki
    from enoki import *
    from enoki.cuda_autodiff import Vector3f as Vector3fD
    from enoki.cuda import Vector3f as Vector3fC


class RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, integrator, sc, param):
        # update parameters
        sc.param_map['BSDF[0]'].reflectance.data = Vector3fD(param.reshape(-1, 3))
        sc.configure2([0])

        psdr_image = integrator.renderC(sc, 0)
        image = psdr_image.torch().to('cuda').to(torch.float32)
        ctx.scene = sc
        ctx.integrator = integrator
        ctx.param = param
        return image.reshape((sc.opts.width, sc.opts.height, 3))

    @staticmethod
    def backward(ctx, grad_out):
        drjit_param = ctx.scene.param_map['BSDF[0]'].reflectance.data
        if test_psdrjit:
            drjit.enable_grad(drjit_param)
        else:
            enoki.set_requires_gradient(drjit_param)
        ctx.scene.configure2([0])
        image_grad = Vector3fC(grad_out.reshape(-1, 3))
        image = ctx.integrator.renderD(ctx.scene, 0)    
        if test_psdrjit:
            tmp = drjit.dot(image_grad, image)
            drjit.backward(tmp)
            grad_tmp = drjit.grad(drjit_param)
        else:
            tmp = enoki.dot(image_grad, image)
            enoki.backward(tmp)
            grad_tmp = enoki.gradient(drjit_param)
        param_grad = torch.nan_to_num(grad_tmp.torch().cuda())
        return tuple([None] * 2 + [param_grad])

class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, integrator, scene, param):
        image = RenderFunction.apply(integrator, scene, param)
        return image


output_path = Path('test_inv2')
output_path.mkdir(parents=True, exist_ok=True)
sc = psdr.Scene()
sc.load_file("bunny_env.xml")
num_pixels = sc.opts.width * sc.opts.height
integrator = psdr.PathTracer(1)	


# Write target image
sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0
opt_target = torch.tensor([0.8, 0.2, 0.9]).to('cuda').to(torch.float32)
render = Renderer()
img_target = render(integrator, sc, opt_target)
write_exr(str(output_path / f"inv_target.exr"), img_target)

opt_param =  torch.tensor([0.5, 0.5, 0.5]).to('cuda').to(torch.float32).requires_grad_()
img_init = render(integrator, sc, opt_param)
write_exr(str(output_path / f"inv_init.exr"), img_init)
optimizer = torch.optim.Adam([opt_param], lr=0.05)
# Optimization
sc.opts.spp = 8
num_iter = 100
for it in range(0, num_iter):
    t0 = time.process_time()
    optimizer.zero_grad()
    image = render(integrator, sc, opt_param)
    loss = (img_target - image).abs().mean()
    loss.backward()
    optimizer.step()
    t1 = time.process_time()
    if test_psdrjit:
        write_exr(str(output_path) + "/drjit_%d.exr" % it, image)
    else:
        write_exr(str(output_path) + "/enoki_%d.exr" % it, image)
    print("iter = %i, loss = %.4e, time = %f sec" % (it, loss.item(), t1-t0))

