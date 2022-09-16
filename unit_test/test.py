import cv2

import sys

def test_diff():
	print("INIT test diff")
	sc = psdr.Scene()
	sc.load_file("cbox.xml")
	sc.configure()
	integrator = psdr.PathTracer(1)	

	P = FloatD(0.)
	drjit.enable_grad(P)

	sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,P],
													 [0.,1.,0.,0.],
													 [0.,0.,1.,0.],
													 [0.,0.,0.,1.],]))
	sc.configure()
	img = integrator.renderD(sc, 0)

	drjit.set_grad(P, 100.0)

	drjit.forward_to(img)
	diff_img = drjit.grad(img)
	diff_img = diff_img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
	cv2.imwrite("psdr_jit_diff_debug.exr", output)


def test_scene():
	print("INIT test scene")
	sc = psdr.Scene()

	sc.load_file("cbox.xml")

	# sc.load_file("bunny_env.xml", False)

	sc.configure()
	integrator = psdr.PathTracer(3)
	# integrator = psdr.CollocatedIntegrator(1000000)

	npass = 2
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
		from drjit.cuda.ad import Float as FloatD, Matrix4f as Matrix4fD

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

	# test_scene()
	test_diff()
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