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
sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.log_level = 0

# sc.param_map["BSDF[0]"].reflectance.to_world = Matrix3fD([[1.,0.,0.2],[0.,1.,0.2],[0.,0.,1.]])
# print(sc.param_map["BSDF[0]"].reflectance.resolution, sc.param_map["BSDF[0]"].reflectance.data)


tex_init=[10.,.05,.05,1.2,1.2]


rot_T = torch.tensor(tex_init[0], device='cuda').requires_grad_()
tran_T = torch.tensor([tex_init[1], tex_init[2]], device='cuda').requires_grad_()
scale_T = torch.tensor([tex_init[3], tex_init[4]], device='cuda').requires_grad_()


tex_to_world = torch.matmul(torch.matmul(rotateT(rot_T), translateT(tran_T)), scaleT(scale_T))
# tex_to_world = translateT(tran_T)
sc.param_map["BSDF[0]"].reflectance.set_transform(Matrix3fD(tex_to_world.detach().cpu().numpy()))

sc.configure()


col_integrator = psdr.CollocatedIntegrator(200)	

# Write target image
img_target = col_integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))




output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"target.exr"), output)


# inital_map = np.zeros((sc.param_map["BSDF[0]"].reflectance.resolution[0],sc.param_map["BSDF[0]"].reflectance.resolution[1],3))+np.array([.5,.5,.5])
# inital_map = inital_map.reshape(-1,3)
# sc.param_map["BSDF[0]"].reflectance.data = Vector3fD(inital_map)

tex_to_world = torch.eye(3, device='cuda', dtype=torch.float32)
sc.param_map["BSDF[0]"].reflectance.set_transform(Matrix3fD(tex_to_world.detach().cpu().numpy()))


sc.configure([0], True)

# Write target image
curr_image = col_integrator.renderC(sc, 0)
img = curr_image.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"inital.exr"), output)

# exit()

class RenderFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, integrator, scene, sensor_id, param):
		# scene.param_map["BSDF[0]"].reflectance.data = Vector3fD(param.detach().cpu().numpy().reshape(-1,3))
		sc.param_map["BSDF[0]"].reflectance.set_transform(Matrix3fD(param.detach().cpu().numpy()))


		scene.configure([0], True)
		psdr_image = integrator.renderC(scene, 0)
		image = psdr_image.torch()
		ctx.scene = scene
		ctx.integrator = integrator
		ctx.param = param
		return image.reshape((scene.opts.height, scene.opts.width, 3))

	@staticmethod
	def backward(ctx, grad_out):
		drjit_param = ctx.scene.param_map["BSDF[0]"].reflectance.to_world
		drjit.enable_grad(drjit_param)
		ctx.scene.configure([0], True)
		image_grad = Vector3fC(grad_out.reshape(-1,3))

		image = ctx.integrator.renderD(ctx.scene, 0)
		tmp = drjit.dot(image_grad, image)
		drjit.backward(tmp)
		grad_tmp = drjit.grad(drjit_param)
		# print(grad_tmp)
		param_grad = torch.nan_to_num(grad_tmp.torch().cuda())
		return tuple([None]*3 + [param_grad])

class Renderer(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, integrator, scene, sensor_id, param):
		image = RenderFunction.apply(integrator,scene,sensor_id,param)
		return image

psdr_render = Renderer()


# opt_map = np.zeros((sc.param_map["BSDF[0]"].reflectance.resolution[0],sc.param_map["BSDF[0]"].reflectance.resolution[1],3))+np.array([.5,.5,.5])
# opt_map = torch.tensor(opt_map, device="cuda", dtype=torch.float32).requires_grad_()
# tex_to_world = torch.matmul(torch.matmul(rotateT(rot_T), translateT(tran_T)), scaleT(scale_T))

# exit()
# optimizer = torch.optim.Adam([{'params':opt_map, "lr":0.01}])
# optimizer = torch.optim.Adam([{'params':rot_T, "lr":0.0}])
# optimizer.add_param_group({'params':tran_T, "lr":0.0})
# optimizer.add_param_group({'params':scale_T, "lr":0.0})

rot_T = torch.tensor(0., device='cuda').requires_grad_()
tran_T = torch.tensor([0.,0.], device='cuda').requires_grad_()
scale_T = torch.tensor([1.,1.], device='cuda').requires_grad_()


optimizer = torch.optim.Adam([{'params':tran_T, "lr":0.01}])
optimizer.add_param_group({'params':rot_T, "lr":0.5})
optimizer.add_param_group({'params':scale_T, "lr":0.1})


target_img = img_target.torch().reshape((sc.opts.width, sc.opts.height, 3))
num_iter = 10000
for it in range(num_iter):
	optimizer.zero_grad()


	tex_to_world = torch.matmul(torch.matmul(rotateT(rot_T), translateT(tran_T)), scaleT(scale_T))

	curr_img = psdr_render(col_integrator, sc, 0, tex_to_world)
	# print(curr_img.shape)
	# print(target_img.shape)
	loss = (target_img-curr_img).abs().mean()
	loss.backward()

	print("rot_T", rot_T)
	print("tran_T", tran_T)
	print("scale_T", scale_T)

	optimizer.step()
	print("it:", it, "loss", loss.item())
	img = curr_img.detach().cpu().numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# cv2.imwrite(str(output_path / f"iter_{it}.exr"), output)

	del curr_img, loss

	# exit()