import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path


test_psdrjit = int(sys.argv[1])

if test_psdrjit:
	import psdr_jit as psdr
	import drjit
	# drjit.set_flag(drjit.JitFlag.PrintIR, True)
	from drjit.cuda.ad import Array3f as Vector3fD
else:
	import psdr_cuda as psdr
	import enoki
	from enoki import *
	from enoki.cuda_autodiff import Vector3f as Vector3fD


output_path = Path('test_inv')
output_path.mkdir(parents=True, exist_ok=True)

sc = psdr.Scene()
sc.load_file("bunny_env.xml")
sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.log_level = 0
new_diffuse = Vector3fD([0.8, 0.2, 0.9])
sc.param_map['BSDF[0]'].reflectance.data = new_diffuse
sc.configure()

integrator = psdr.PathTracer(1)	

# Write target image
img_target = integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))
num_pixels = sc.opts.width * sc.opts.height
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"inv_target.exr"), output)

# Write initial image
init_diffuse = Vector3fD([0.5, 0.5, 0.5])
if test_psdrjit:
	drjit.enable_grad(init_diffuse)
else:
	enoki.set_requires_gradient(init_diffuse, True)
sc.param_map['BSDF[0]'].reflectance.data = init_diffuse
sc.configure()
img = integrator.renderC(sc, 0)
img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path /"inv_init.exr"), output)

sc.opts.spp = 8
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0

num_iter = 100
for it in range(0, num_iter):
	t0 = time.process_time()
	# if it == num_iter - 1:
	# 	drjit.set_flag(drjit.JitFlag.PrintIR, True)
	if test_psdrjit:
		init_diffuse = Vector3fD(drjit.detach(init_diffuse))
		drjit.enable_grad(init_diffuse)
	else:
		init_diffuse = Vector3fD(enoki.detach(init_diffuse))
		enoki.set_requires_gradient(init_diffuse, True)
	sc.param_map['BSDF[0]'].reflectance.data = init_diffuse
	sc.configure()
	img = integrator.renderD(sc, 0)
	if test_psdrjit:
		loss = drjit.mean(drjit.mean(drjit.abs(img_target - img)))
		drjit.backward(loss)
		texture_g = drjit.grad(init_diffuse)
		if drjit.none(drjit.isnan(texture_g))[0]:
			init_diffuse = init_diffuse-texture_g
	else:
		loss = 0
		for i in range(3):  
			I_ = img[i]
			T_ = img_target[i]
			I = enoki.select(I_ > 1, 1.0, I_)
			T = enoki.select(T_ > 1, 1.0, T_)
			diff = enoki.abs(I - T)
			loss += enoki.hsum(enoki.hsum(diff)) / (num_pixels * 3)
		enoki.backward(loss)
		texture_g = enoki.gradient(init_diffuse)
		if enoki.none(enoki.isnan(texture_g))[0]:
			init_diffuse = init_diffuse-texture_g
	t1 = time.process_time()
	output = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
	if test_psdrjit:
		cv2.imwrite(str(output_path) + "/drjit_%d.exr" % it, output)
		drjit.eval(init_diffuse)
		# drjit.set_flag(drjit.JitFlag.PrintIR, False)
	else:
		cv2.imwrite(str(output_path) + "/enoki_%d.exr" % it, output)
	print("iter = %i, loss = %.4e, time = %f sec" % (it, loss.numpy().item(), t1-t0))





