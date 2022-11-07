import cv2

import sys
import torch



def test_diff():
	print("INIT test diff")
	sc = psdr.Scene()
	sc.load_file("cbox.xml")
	sc.opts.spp = 32
	sc.opts.sppe = 32
	sc.opts.sppse = 32
	sc.configure()
	integrator = psdr.PathTracer(3)	

	P = FloatD(0.)

	if test_psdrjit:
		drjit.enable_grad(P)
	else:
		enoki.set_requires_gradient(P)
	
	sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,P*100.],
													 [0.,1.,0.,0.],
													 [0.,0.,1.,0.],
													 [0.,0.,0.,1.],]))
	sc.configure()
	img = integrator.renderD(sc, 0)

	if test_psdrjit:
		drjit.set_grad(P, 1.0)
		drjit.forward_to(img)
		diff_img = drjit.grad(img)
		diff_img = diff_img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
		output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
		cv2.imwrite("psdr_jit_diff_debug.exr", output)
	else:
		enoki.forward(P, free_graph=True)
		diff_img = enoki.gradient(img)
		diff_img = diff_img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
		output = cv2.cvtColor(diff_img, cv2.COLOR_RGB2BGR)
		cv2.imwrite("psdr_cuda_diff_debug.exr", output)

def test_load_scene():
	print("INIT load scene")
	sc = psdr.Scene()
	sc.opts.spp = 32
	sc.opts.sppe = 32
	sc.opts.sppse = 32
	sc.opts.height = 512
	sc.opts.width = 512

	integrator = psdr.PathTracer(3)	


	sensor = psdr.PerspectiveCamera(60, 0.000001, 10000000.)
	to_world = Matrix4fD([[1.,0.,0.,208.],
						 [0.,1.,0.,273.],
						 [0.,0.,1.,-800.],
						 [0.,0.,0.,1.],])
	sensor.to_world = to_world
	sc.add_Sensor(sensor)

	bsdf = psdr.DiffuseBSDF()
	sc.add_BSDF(psdr.DiffuseBSDF([0.0, 0.0, 0.0]), "light")
	sc.add_BSDF(psdr.DiffuseBSDF(), "cat")
	sc.add_BSDF(psdr.DiffuseBSDF([0.95, 0.95, 0.95]), "white")
	sc.add_BSDF(psdr.DiffuseBSDF([0.20, 0.90, 0.20]), "green")
	sc.add_BSDF(psdr.DiffuseBSDF([0.90, 0.20, 0.20]), "red")

	sc.add_Mesh("./data/objects/cbox/cbox_luminaire.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,-0.5],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "light", psdr.AreaLight([20.0, 20.0, 8.0]))
	sc.add_Mesh("./data/objects/cbox/cbox_smallbox.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "cat", None)
	sc.add_Mesh("./data/objects/cbox/cbox_largebox.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "cat", None)
	sc.add_Mesh("./data/objects/cbox/cbox_floor.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
	sc.add_Mesh("./data/objects/cbox/cbox_ceiling.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
	sc.add_Mesh("./data/objects/cbox/cbox_back.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "white", None)
	sc.add_Mesh("./data/objects/cbox/cbox_greenwall.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "green", None)
	sc.add_Mesh("./data/objects/cbox/cbox_redwall.obj", Matrix4fC([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]), "red", None)

	P = FloatD(0.)
	drjit.enable_grad(P)
	
	sc.param_map["Mesh[0]"].set_transform(Matrix4fD([[1.,0.,0.,P*100.],
													 [0.,1.,0.,0.],
													 [0.,0.,1.,0.],
													 [0.,0.,0.,1.],]))


	sc.configure()
	img = integrator.renderD(sc, 0)
	org_img = img.numpy().reshape((sc.opts.width, sc.opts.height, 3))
	output = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
	cv2.imwrite("psdr_jit_forward.exr", output)


	drjit.set_grad(P, 1.0)
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
	for i in range(0, 1000):
		val = sampler.next_1d();
		drjit.eval(val)
		print("iter", i)

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
		from drjit.cuda import Float as FloatC, Matrix4f as Matrix4fC

		print("testing psdr-jit")
	else:
		import psdr_cuda as psdr
		import enoki
		from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD

		print("testing psdr-cuda")


	# test_drjit_init()
	# test_sampler()
	# test_ray()
	# test_DiscreteDistribution()
	# test_mesh()

	# test_scene()
	# test_diff()
	test_load_scene()
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