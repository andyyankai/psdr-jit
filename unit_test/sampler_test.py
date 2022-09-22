import sys

def test_sampler():
	print("testing psdr_jit::Sampler")
	drjit.set_flag(drjit.JitFlag.LoopRecord, False)

	sampler = psdr.Sampler()
	sampler.seed([1,4,3])
	for i in range(0, 1000):
		val = sampler.next_2d();
		if test_psdrjit:
			drjit.eval(val)
			print(val)
		else:
			print(val)
		print("iter", i)

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
		from enoki.cuda_autodiff import Float32 as FloatD, Vector3f as Vector3fD, Matrix4f as Matrix4fD

		print("testing psdr-cuda")

	test_sampler()


	if test_psdrjit:
		print("FINISH psdr-jit")
	else:
		print("FINISH psdr-cuda")
