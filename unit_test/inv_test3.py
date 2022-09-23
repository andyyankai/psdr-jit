from ast import Delete
import os
import time
from xmlrpc.client import Boolean
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path
import torch
import numpy as np

test_psdrjit = int(sys.argv[1])
rnd_cameras = True

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

t_res = 1024

class RenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, integrator, sc, sensor_ids, *params):
        # update parameters
        sc.param_map['BSDF[0]'].diffuseReflectance.data = Vector3fD(params[0].reshape(-1, 3))
        sc.param_map['BSDF[0]'].specularReflectance.data = Vector3fD(params[1].reshape(-1, 3))
        if torch.is_tensor(sensor_ids):
            sensor_ids = sensor_ids.flatten().tolist()
        sc.configure2(sensor_ids)
        images = []
        for sensor_id in sensor_ids:
            psdr_image = integrator.renderC(sc, sensor_id).torch().to('cuda').to(torch.float32)
            images.append(psdr_image.reshape((sc.opts.height, sc.opts.width, 3)))
        concated_image = torch.cat(images, dim=1)
        ctx.scene = sc
        ctx.integrator = integrator
        ctx.sensor_ids = sensor_ids
        return concated_image

    @staticmethod
    def backward(ctx, grad_out):
        drjit_params = [ctx.scene.param_map['BSDF[0]'].diffuseReflectance.data,
                        ctx.scene.param_map['BSDF[0]'].specularReflectance.data]
                        # ctx.scene.param_map['BSDF[0]'].roughness.data]
        for drjit_param in drjit_params:
            if test_psdrjit:
                drjit.enable_grad(drjit_param)           
            else:
                enoki.set_requires_gradient(drjit_param)
        ctx.scene.configure2(ctx.sensor_ids)
        image_grad = Vector3fC(grad_out.reshape(-1, 3))
        param_grads = [torch.zeros_like(drjit_param.torch().cuda()) for drjit_param in drjit_params]
        for sensor_id in ctx.sensor_ids:
            image = ctx.integrator.renderD(ctx.scene, sensor_id)
            tmp = dot(image_grad, image)
            backward(tmp)
            for idx_param in range(len(drjit_params)):
                param_grads[idx_param] += torch.nan_to_num(grad(drjit_params[idx_param]).torch().cuda())
        
        for idx_param in range(len(drjit_params)):
            param_grads[idx_param] = param_grads[idx_param].reshape(t_res, t_res, -1)
        return tuple([None] * 3 + param_grads)

class Renderer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, integrator, scene, sensor_id, params = []):
        image = RenderFunction.apply(integrator, scene, sensor_id, *params)
        return image


output_path = Path('test_inv3')
output_path.mkdir(parents=True, exist_ok=True)
integrator = psdr.PathTracer(1)
vis_sensor_id = [210, 18, 267, 209]

# load scene
old_path = os.getcwd()
tmp_path = Path("data/butterfly_flower")
sc = psdr.Scene()
os.chdir(tmp_path)
sc.load_file("scene.xml", False)
os.chdir(old_path)
num_pixels = sc.opts.width * sc.opts.height
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0


# load target image
tar_images = []
target_images_path = Path(r'.\data\butterfly_flower\tar_images')
for target_image_path in sorted(target_images_path.glob('*.exr'), key=lambda x : int(x.stem)):
    img = torch.from_numpy(cv2.imread(str(target_image_path), -1))
    tar_images.append(img.to('cuda').to(torch.float32))

# optimization
D_initial = torch.ones(t_res, t_res, 3).to('cuda').to(torch.float32) * 0.5
D_data = D_initial.requires_grad_()
S_initial = torch.ones(t_res, t_res, 3).to('cuda').to(torch.float32) * 0.04
S_data = S_initial.requires_grad_()
# R_initial = torch.ones(t_res, t_res, 1).to('cuda').to(torch.float32) * 0.5
# R_data = R_initial.requires_grad_()
optimizer = torch.optim.Adam([D_data, S_data], lr=0.1)
render_opt = Renderer()         # for optimization
render_vis = Renderer()         # for visualization

sc.opts.spp = 128
# Optional (write init image)
img_init = render_vis(integrator, sc, vis_sensor_id, [D_data, S_data])
img_init = torch.from_numpy(cv2.cvtColor(img_init.detach().to('cpu').numpy(), cv2.COLOR_RGB2BGR))
img_target = torch.cat([tar_images[id] for id in vis_sensor_id], dim = 1)
img_vis = torch.cat([img_target.to('cpu'), img_init], dim = 0)
cv2.imwrite(str(output_path) + "/inv_init.exr", img_vis.numpy())

# Optimization
num_epochs = 500
batch_size = 1
num_opt_sensors = len(tar_images)
opt_sensor_ids = torch.tensor(list(range(num_opt_sensors)))
iter = 0
sc.opts.spp = 64
for epoch in range(1, num_epochs + 1):
    if rnd_cameras:
        sensor_perm = opt_sensor_ids[torch.randperm(num_opt_sensors)]
        # sensor_perm = opt_sensor_ids[torch.randperm(20)]
        sensor_batches = torch.split(sensor_perm, batch_size)
    else:
        sensor_batches = (torch.tensor([0]), torch.tensor([1]))
    for sensor_ids in sensor_batches:
        optimizer.zero_grad()
        t0 = time.process_time()
        image = render_opt(integrator, sc, sensor_ids, [D_data, S_data])
        img_target = torch.cat([tar_images[id] for id in sensor_ids], dim = 1)  
        loss = (img_target - image).abs().mean()
        loss.backward()
        optimizer.step()
        iter += 1
        t1 = time.process_time()
        print(f"[Epoch {epoch}/{num_epochs}] iter: {iter} | loss: {loss.item()} | time: {t1 - t0}")
        image_vis = torch.cat([image.detach(), img_target], dim = 1)
        cv2.imwrite(str(output_path) + "/ep%d_it%d.exr" % (epoch, iter), img_vis.to('cpu').numpy())
