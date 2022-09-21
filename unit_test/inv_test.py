import cv2

import sys

import psdr_jit as psdr
import drjit
from drjit import *
from drjit.cuda.ad import Float as FloatD, Array3f as Vector3fD, Matrix4f as Matrix4fD
from pathlib import Path

output_path = Path('inv_test')
output_path.mkdir(parents=True, exist_ok=True)

sc = psdr.Scene()
sc.load_file("bunny_env.xml")
sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.log_level = 0

new_diffuse = Vector3fD([0.8, 0.2, 0.9])
sc.param_map['BSDF[0]'].reflectance.data = new_diffuse;


sc.configure()
integrator = psdr.PathTracer(1)	


img_target = integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("inv_target.exr", output)

init_diffuse = Vector3fD([0.5, 0.5, 0.5])
drjit.enable_grad(init_diffuse)

sc.param_map['BSDF[0]'].reflectance.data = init_diffuse;
sc.configure()

img = integrator.renderC(sc, 0)
img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite("inv_init.exr", output)


sc.opts.spp = 32
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0

for it in range(0, 1000):
	init_diffuse = Vector3fD(detach(init_diffuse))
	drjit.enable_grad(init_diffuse)
	sc.param_map['BSDF[0]'].reflectance.data = init_diffuse;
	sc.configure()
	img = integrator.renderD(sc, 0)
	loss = mean(mean(abs(img_target - img)))
	backward(loss)
	texture_g = grad(init_diffuse)
	if none(isnan(texture_g))[0]:
		init_diffuse = init_diffuse-texture_g
	print("iter", it, loss, init_diffuse)
	# print("iter", it)


# loop_spp = 2
# for it in range(0, 100):
# 	init_diffuse = Vector3fD(detach(init_diffuse))
# 	drjit.enable_grad(init_diffuse)
# 	sc.param_map['BSDF[0]'].reflectance.data = init_diffuse;
# 	sc.configure()
# 	I = Vector3fD(integrator.renderC(sc, 0))
# 	drjit.enable_grad(I)
# 	I0 = img_target
# 	L = mean(mean(abs(I0 - I)))
# 	backward(L)
# 	dL = grad(I)
# 	for ii in range(0, loop_spp):
# 		img_ad = integrator.renderD(sc, 0)
# 		tmp = dot(dL/float(loop_spp), img_ad)
# 		backward(tmp)
# 	texture_g = grad(init_diffuse)
# 	init_diffuse = init_diffuse-texture_g
# 	print("iter", it, L, init_diffuse)
# 	# img = I.numpy().reshape((sc.opts.width, sc.opts.height, 3))
# 	# output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# 	# cv2.imwrite(str(output_path)+"/inv_iter"+str(it)+".exr", output)
# 	del texture_g, img_ad, I, tmp, L
# 	drjit.registry_trim()



