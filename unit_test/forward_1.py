import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2, sys
from pathlib import Path


import psdr_jit as psdr
import drjit
# drjit.set_flag(drjit.JitFlag.PrintIR, True)
from drjit.cuda.ad import Array3f as Vector3fD
from drjit.cuda.ad import Float as FloatD, Matrix3f as Matrix3fD


output_path = Path('forward_1')
output_path.mkdir(parents=True, exist_ok=True)

sc = psdr.Scene()
sc.load_file("texture.xml")
sc.opts.spp = 128
sc.opts.sppe = 0
sc.opts.sppse = 0
sc.opts.log_level = 0

sc.param_map["BSDF[0]"].reflectance.to_world = Matrix3fD([[2.5,0.,0.2],[0.,2.5,0.2],[0.,0.,1.]])

sc.configure()

integrator = psdr.CollocatedIntegrator(200)	

# Write target image
img_target = integrator.renderC(sc, 0)
img = img_target.numpy().reshape((sc.opts.width, sc.opts.height, 3))

output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(output_path / f"result.exr"), output)
