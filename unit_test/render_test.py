import cv2

import sys

test_psdrjit = int(sys.argv[1])

if test_psdrjit:
	import psdr_jit as psdr
	import drjit
	from drjit import *
	from drjit.cuda.ad import Float as FloatD, Array3f as Vector3fD, Matrix4f as Matrix4fD
else:
	import psdr_cuda as psdr
	import enoki
	from enoki import *


import time
from pathlib import Path

output_path = Path('inv_test')
output_path.mkdir(parents=True, exist_ok=True)

sc = psdr.Scene()
# sc.load_file("cbox.xml")
sc.load_file("bunny_env.xml")

# integrator = psdr.CollocatedIntegrator(100000)	
# integrator = psdr.PathTracer(3)	
integrator = psdr.FieldExtractionIntegrator("silhouette")
sc.opts.spp = 32
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.sppse = 0
sc.opts.log_level = 0

t0 = time.process_time()

for it in range(0, 1000):
	# init_diffuse = Vector3fD(detach(init_diffuse))
	# drjit.enable_grad(init_diffuse)
	# sc.param_map['BSDF[0]'].reflectance.data = init_diffuse;
	sc.configure()
	img = integrator.renderC(sc, 0)
	# loss = mean(mean(abs(img_target - img)))
	# backward(loss)
	# texture_g = grad(init_diffuse)
	# if none(isnan(texture_g))[0]:
	# 	init_diffuse = init_diffuse-texture_g
	# print("iter", it, loss, init_diffuse)
	# img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	# output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# cv2.imwrite("optix_debug.exr", output)

	print("iter", it)
	# exit()

t1 = time.process_time()

if test_psdrjit:
	print("psdr_jit done in %.2f seconds." % (t1-t0), end="\r")
else:
	print("psdr_cuda done in %.2f seconds." % (t1-t0), end="\r")