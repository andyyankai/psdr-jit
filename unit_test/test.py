test_psdrjit = 1

if test_psdrjit:
	import psdr_jit as psdr
	import drjit
	print("testing psdr-jit")
else:
	import psdr_cuda as psdr
	import enoki
	print("testing psdr-cuda")

def test_drjit_init():
	psdr.drjit_test()


def test_sampler():
	print("testing psdr_jit::Sampler")
	sampler = psdr.Sampler()
	sampler.seed([0,1,2,3])
	print(sampler.next_1d())
	sampler.seed(1)
	print(sampler.next_1d())

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


if __name__ == "__main__":
	test_drjit_init()
	test_sampler()
	test_ray()
	test_DiscreteDistribution()

# print("testing drjit autodiff")
# psdr.drjit_test()

# print("testing psdr_jit::Sampler")
# sampler = psdr.Sampler()
# sampler.seed([1,2,3])
# print(sampler.next_1d())
# sampler.seed(1)
# print(sampler.next_1d())