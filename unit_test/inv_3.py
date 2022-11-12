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


sc.configure()

col_integrator = psdr.CollocatedIntegrator(200)	

# Write target image
img_target = col_integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))

output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"target.exr"), output)


inital_map = np.zeros((sc.param_map["BSDF[0]"].reflectance.resolution[0],sc.param_map["BSDF[0]"].reflectance.resolution[1],3))+np.array([.5,.5,.5])
inital_map = inital_map.reshape(-1,3)
sc.param_map["BSDF[0]"].reflectance.data = Vector3fD(inital_map)

sc.configure([0], True)

# Write target image
curr_image = col_integrator.renderC(sc, 0)
img = curr_image.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"inital.exr"), output)


class RenderFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, integrator, scene, sensor_id, param):
		scene.param_map["BSDF[0]"].reflectance.data = Vector3fD(param.detach().cpu().numpy().reshape(-1,3))
		scene.configure([0], True)
		psdr_image = integrator.renderC(scene, 0)
		image = psdr_image.torch()
		ctx.scene = scene
		ctx.integrator = integrator
		ctx.param = param
		return image.reshape((scene.opts.height, scene.opts.width, 3))

	@staticmethod
	def backward(ctx, grad_out):
		drjit_param = ctx.scene.param_map["BSDF[0]"].reflectance.data
		drjit.enable_grad(drjit_param)
		ctx.scene.configure([0], True)
		image_grad = Vector3fC(grad_out.reshape(-1,3))

		image = ctx.integrator.renderD(ctx.scene, 0)
		tmp = drjit.dot(image_grad, image)
		drjit.backward(tmp)
		grad_tmp = drjit.grad(drjit_param)
		param_grad = torch.nan_to_num(grad_tmp.torch().cuda()).reshape(512,512,3)
		return tuple([None]*3 + [param_grad])

class Renderer(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, integrator, scene, sensor_id, param):
		image = RenderFunction.apply(integrator,scene,sensor_id,param)
		return image

psdr_render = Renderer()


opt_map = np.zeros((sc.param_map["BSDF[0]"].reflectance.resolution[0],sc.param_map["BSDF[0]"].reflectance.resolution[1],3))+np.array([.5,.5,.5])
opt_map = torch.tensor(opt_map, device="cuda", dtype=torch.float32).requires_grad_()
render = psdr_render(col_integrator, sc, 0, opt_map)

img = render.detach().cpu().numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"renderer.exr"), output)


optimizer = torch.optim.Adam([{'params':opt_map, "lr":0.01}])
target_img = img_target.torch().reshape((sc.opts.width, sc.opts.height, 3))
num_iter = 100
for it in range(num_iter):
	optimizer.zero_grad()
	curr_img = psdr_render(col_integrator, sc, 0, opt_map)
	# print(curr_img.shape)
	# print(target_img.shape)
	loss = (target_img-curr_img).abs().mean()
	loss.backward()
	optimizer.step()
	print("it:", it, "loss", loss.item())
	img = curr_img.detach().cpu().numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.imwrite(str(output_path / f"iter_{it}.exr"), output)

	del curr_img, loss

	# exit()