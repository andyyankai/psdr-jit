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
from drjit.cuda.ad import Array3f as Vector3fD, Array2f as Vector2fD, Float as FloatD
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


scale_init = 1.5
trans_init = [0.1,-0.2]
rot_init = 0.5

# tex_to_world = torch.matmul(torch.matmul(rotateT(rot_T), translateT(tran_T)), scaleT(scale_T))
# tex_to_world = translateT(tran_T)
sc.param_map["BSDF[0]"].reflectance.scale = FloatD(scale_init)
sc.param_map["BSDF[0]"].reflectance.translate = Vector2fD(trans_init)
sc.param_map["BSDF[0]"].reflectance.rotate = FloatD(rot_init)

sc.configure()


col_integrator = psdr.CollocatedIntegrator(200)	

# Write target image
img_target = col_integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))




output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"target.exr"), output)
# exit()

# inital_map = np.zeros((sc.param_map["BSDF[0]"].reflectance.resolution[0],sc.param_map["BSDF[0]"].reflectance.resolution[1],3))+np.array([.5,.5,.5])
# inital_map = inital_map.reshape(-1,3)
# sc.param_map["BSDF[0]"].reflectance.data = Vector3fD(inital_map)


sc.param_map["BSDF[0]"].reflectance.scale = FloatD(1.0)
sc.param_map["BSDF[0]"].reflectance.translate = Vector2fD(0.0, 0.0)
sc.param_map["BSDF[0]"].reflectance.rotate = FloatD(0.0)


sc.configure([0], True)

# Write target image
curr_image = col_integrator.renderC(sc, 0)
img = curr_image.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"inital.exr"), output)

# exit()

class RenderFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, integrator, scene, sensor_id, param1, param2, param3):

		sc.param_map["BSDF[0]"].reflectance.scale = FloatD(param1)
		sc.param_map["BSDF[0]"].reflectance.translate = Vector2fD(param2)
		sc.param_map["BSDF[0]"].reflectance.rotate = FloatD(param3)

		scene.configure([0], True)
		psdr_image = integrator.renderC(scene, 0)
		image = psdr_image.torch()
		ctx.scene = scene
		ctx.integrator = integrator
		ctx.param1 = param1
		ctx.param2 = param2
		ctx.param3 = param3
		return image.reshape((scene.opts.height, scene.opts.width, 3))

	@staticmethod
	def backward(ctx, grad_out):
		drjit_param1 = ctx.scene.param_map["BSDF[0]"].reflectance.scale
		drjit.enable_grad(drjit_param1)

		drjit_param2 = ctx.scene.param_map["BSDF[0]"].reflectance.translate
		drjit.enable_grad(drjit_param2)

		drjit_param3 = ctx.scene.param_map["BSDF[0]"].reflectance.rotate
		drjit.enable_grad(drjit_param3)



		ctx.scene.configure([0], True)
		image_grad = Vector3fC(grad_out.reshape(-1,3))

		image = ctx.integrator.renderD(ctx.scene, 0)
		tmp = drjit.dot(image_grad, image)
		drjit.backward(tmp)

		param_grad1 = torch.nan_to_num(drjit.grad(drjit_param1).torch().cuda())
		param_grad2 = torch.nan_to_num(drjit.grad(drjit_param2).torch().cuda())
		param_grad3 = torch.nan_to_num(drjit.grad(drjit_param3).torch().cuda())
		return tuple([None]*3 + [param_grad1] + [param_grad2] + [param_grad3])

class Renderer(torch.nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, integrator, scene, sensor_id, param1, param2, param3):
		image = RenderFunction.apply(integrator,scene,sensor_id,param1, param2, param3)
		return image

psdr_render = Renderer()



opt_scale = torch.tensor([1.0], device='cuda', dtype=torch.float32).requires_grad_()
opt_trans = torch.tensor([[0.0,0.0]], device='cuda', dtype=torch.float32).requires_grad_()
opt_rot = torch.tensor([0.0], device='cuda', dtype=torch.float32).requires_grad_()

optimizer = torch.optim.Adam([{'params':opt_scale, "lr":0.01},{'params':opt_trans, "lr":0.01},{'params':opt_rot, "lr":0.01}])

target_img = img_target.torch().reshape((sc.opts.width, sc.opts.height, 3))
num_iter = 100
for it in range(num_iter):
	optimizer.zero_grad()


	curr_img = psdr_render(col_integrator, sc, 0, opt_scale, opt_trans, opt_rot)
	# print(curr_img.shape)
	# print(target_img.shape)
	loss = (target_img-curr_img).abs().mean()
	loss.backward()

	print("opt_scale", opt_scale)
	print("opt_trans", opt_trans)
	print("opt_rot", opt_rot)

	optimizer.step()
	print("it:", it, "loss", loss.item())
	img = curr_img.detach().cpu().numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	cv2.imwrite(str(output_path / f"iter_{it}.exr"), output)

	del curr_img, loss

	# exit()