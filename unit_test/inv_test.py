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
sc.opts.spp = 32
sc.opts.sppe = 0
sc.opts.sppse = 0
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


for it in range(0, 100):
	init_diffuse = Vector3fD(detach(init_diffuse))
	drjit.enable_grad(init_diffuse)
	sc.param_map['BSDF[0]'].reflectance.data = init_diffuse;
	sc.configure()
	img = integrator.renderD(sc, 0)
	loss = mean(mean(abs(img_target - img)))
	backward(loss)
	texture_g = grad(init_diffuse)
	init_diffuse = init_diffuse-texture_g
	print("iter", it, loss, init_diffuse)

	# img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	# output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# cv2.imwrite(str(output_path)+"/inv_iter"+str(it)+".exr", output)


