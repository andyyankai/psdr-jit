import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path
from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD


test_psdrjit = 1

import psdr_jit as psdr
import drjit
# drjit.set_flag(drjit.JitFlag.PrintIR, True)
from drjit.cuda.ad import Array3f as Vector3fD


output_path = Path('result','inv_2')
output_path.mkdir(parents=True, exist_ok=True)

sc = psdr.Scene()
sc.load_file("bunny_env.xml")
sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
# sc.opts.log_level = 0


print(sc.param_map['Mesh[0]'].to_world)


sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,0.],
												 [0.,1.,0.,100.],
												 [0.,0.,1.,0.],
												 [0.,0.,0.,1.],]))

sc.configure([0],True)

integrator = psdr.PathTracer(1)	

# Write target image
img_target = integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))
num_pixels = sc.opts.width * sc.opts.height
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"inv_target.exr"), output)


# exit()
# Write initial image
init_pos = FloatD(0.0)

drjit.enable_grad(init_pos)

sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,0.],
												 [0.,1.,0.,init_pos],
												 [0.,0.,1.,0.],
												 [0.,0.,0.,1.],]))

sc.configure([0],True)
img = integrator.renderC(sc, 0)
img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path /"inv_init.exr"), output)

sc.opts.spp = 8
sc.opts.sppe = 8
sc.opts.sppse = 0
sc.opts.log_level = 0

num_iter = 100
for it in range(0, num_iter):
	t0 = time.process_time()
	init_pos = FloatD(drjit.detach(init_pos))

	# print(init_pos)
	drjit.enable_grad(init_pos)

	sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,0.],
													 [0.,1.,0.,init_pos],
													 [0.,0.,1.,0.],
													 [0.,0.,0.,1.],]))

	sc.configure([0],True)
	# print("config")
	img = integrator.renderD(sc, 0)
	# drjit.eval(img)
	# print(img)
	# print("render")

	loss = drjit.mean(drjit.mean(drjit.abs(img_target - img)))

	drjit.backward(loss)

	curr_pos = drjit.grad(init_pos)

	if drjit.none(drjit.isnan(curr_pos)):
		init_pos = init_pos-curr_pos*10000.

	t1 = time.process_time()
	output = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

	cv2.imwrite(str(output_path) + "/drjit_%d.exr" % it, output)

	print("iter = %i, loss = %.4e, time = %f sec" % (it, loss.numpy().item(), t1-t0))

	# exit()






