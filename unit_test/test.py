import cv2

import sys


def test_scene():
	print("INIT test scene")
	sc = psdr.Scene()

	sc.load_file("cbox.xml", False)

	# sc.load_file("bunny_env.xml")

	sc.configure()
	integrator = psdr.PathTracer(1)

	npass = 10
	for n in range(npass):
		if n==0:
			img = integrator.renderC(sc, 0)
		else:
			img += integrator.renderC(sc, 0)

	img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3)) / npass
	output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


	if test_psdrjit:
		cv2.imwrite("psdr_jit_debug.exr", output)
	else:
		cv2.imwrite("psdr_cuda_debug.exr", output)


	img = integrator.renderC(sc, 0)
	# print(img)
	img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	# print(img)
	# print()
	output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


	if test_psdrjit:
		cv2.imwrite("psdr_jit_debug2.exr", output)
	else:
		cv2.imwrite("psdr_cuda_debug2.exr", output)


	print("FIN test scene")


def test_drjit_init():
	psdr.drjit_test()


def test_sampler():
	print("testing psdr_jit::Sampler")

	sampler = psdr.Sampler()
	sampler.seed([1,2,3])
	print(sampler.next_1d())
	print(sampler.next_1d())


	# sampler = psdr.Sampler()
	# sampler.seed([0,1,2,3])
	# print(sampler.next_1d())
	# sampler.seed(1)
	# print(sampler.next_1d())

def test_ray():
	print("test ray")
	rayc = psdr.RayC([0.,0.,0.], [1.,0.,0.])
	print(rayc.o)
	print(rayc.d)

	rayd = psdr.RayD([0.,0.,0.], [1.,0.,0.])
	print(rayd.o)
	print(rayd.d)

def test_DiscreteDistribution():
	print("test DiscreteDistribution")

	sampler = psdr.Sampler()
	sampler.seed([0,1,2,3])


	distrb = psdr.DiscreteDistribution()
	distrb.init([1.2,2.1,1.4,0.4])
	print(distrb.pmf())
	print(distrb.sum)
	print(distrb.sample(sampler.next_1d()))


def test_mesh():
	print("before mesh")
	mesh = psdr.Mesh()
	mesh.load("cube_uv.obj", True)
	print("load")
	mesh.configure()
	print(mesh.to_world)
	print("dump")
	mesh.dump("out.obj")
	print("after mesh")

if __name__ == "__main__":
	print("psdr_cuda test: python test.py 0")
	print("psdr_jit test: python test.py 1")

	test_psdrjit = int(sys.argv[1])
	if test_psdrjit:
		import psdr_jit as psdr
		import drjit
		print("testing psdr-jit")
	else:
		import psdr_cuda as psdr
		import enoki
		print("testing psdr-cuda")


	# test_drjit_init()
	# test_sampler()
	# test_ray()
	# test_DiscreteDistribution()
	# test_mesh()

	test_scene()

	# psdr.drjit_memory()


	if test_psdrjit:
		print("FINISH psdr-jit")
	else:
		print("FINISH psdr-cuda")


	# print(psdr.IntersectionD)

# print("testing drjit autodiff")
# psdr.drjit_test()

# # print("testing psdr_jit::Sampler")
# sampler = psdr.Sampler()
# # sampler.seed([1,2,3])
# print(sampler.next_1d())
# # print(sampler.next_1d())