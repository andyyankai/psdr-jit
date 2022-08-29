

# import psdr_jit as psdr
# import drjit
import psdr_cuda as psdr
import enoki

print("running test on psdr_jit")
print("testing drjit autodiff")

psdr.drjit_test()

print("testing psdr_jit::Sampler")

sampler = psdr.Sampler()
sampler.seed([1,2,3])
print(sampler.next_1d())
sampler.seed(1)
print(sampler.next_1d())