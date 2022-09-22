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

t_res = 1024

class RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, integrator, sc, sensor_id, param):
        # update parameters
        sc.param_map['BSDF[0]'].diffuseReflectance.data = Vector3fD(param.reshape(-1, 3))
        sc.configure2(sensor_id)
        psdr_image = integrator.renderC(sc, sensor_id[0])
        image = psdr_image.torch().to('cuda').to(torch.float32)
        ctx.scene = sc
        ctx.integrator = integrator
        ctx.param = param
        ctx.sensor_id = sensor_id
        return image.reshape((sc.opts.height, sc.opts.width, 3))

    @staticmethod
    def backward(ctx, grad_out):
        drjit_param = ctx.scene.param_map['BSDF[0]'].diffuseReflectance.data
        if test_psdrjit:
            drjit.enable_grad(drjit_param)
        else:
            enoki.set_requires_gradient(drjit_param)
        ctx.scene.configure2([0])
        image_grad = Vector3fC(grad_out.reshape(-1, 3))
        image = ctx.integrator.renderD(ctx.scene, ctx.sensor_id[0])    
        if test_psdrjit:
            tmp = drjit.dot(image_grad, image)
            drjit.backward(tmp)
            grad_tmp = drjit.grad(drjit_param)
        else:
            tmp = enoki.dot(image_grad, image)
            enoki.backward(tmp)
            grad_tmp = enoki.gradient(drjit_param)
        param_grad = torch.nan_to_num(grad_tmp.torch().cuda()).reshape(t_res, t_res, 3)
        return tuple([None] * 3 + [param_grad])

class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, integrator, scene, sensor_id, param):
        image = RenderFunction.apply(integrator, scene, sensor_id, param)
        return image


output_path = Path('test_inv4')
output_path.mkdir(parents=True, exist_ok=True)
integrator = psdr.PathTracer(1)
sensor_id = [0]

# load scene
sc = psdr.Scene()
old_path = os.getcwd()
tmp_path = Path("data/tex_square")
os.chdir(tmp_path)
sc.load_file("scene.xml", False)
os.chdir(old_path)
num_pixels = sc.opts.width * sc.opts.height

sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.log_level = 0
sc.configure()
img = integrator.renderC(sc, 0)
img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
num_pixels = sc.opts.width * sc.opts.height
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"inv_target.exr"), output)
exit()
# load target image
# img_target = read_exr("data/butterfly_flower/tar_images/" + str(sensor_id[0]) + '.exr')
# img_target = torch.from_numpy(img_target).to('cuda').to(torch.float32)
# write_exr(str(output_path / f"inv_target.exr"), img_target)

# optimization
render = Renderer()
initial_value = torch.ones(t_res, t_res, 3).to('cuda').to(torch.float32) * 0.5
D_data = initial_value.requires_grad_()
optimizer = torch.optim.Adam([D_data], lr=0.1)
sc.opts.spp = 32
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0
# Optional (write init image)
img_init = render(integrator, sc, sensor_id, D_data)
write_exr(str(output_path / f"inv_init.exr"), img_init)
exit()
# Optimization
sc.opts.spp = 8
num_iter = 100
for it in range(0, num_iter):
    t0 = time.process_time()
    optimizer.zero_grad()
    image = render(integrator, sc, sensor_id, D_data)
    loss = (img_target - image).abs().mean()
    loss.backward()
    optimizer.step()
    t1 = time.process_time()
    if test_psdrjit:
        write_exr(str(output_path) + "/drjit_%d.exr" % it, image)
    else:
        write_exr(str(output_path) + "/enoki_%d.exr" % it, image)
    print("iter = %i, loss = %.4e, time = %f sec" % (it, loss.item(), t1-t0))
    del loss

