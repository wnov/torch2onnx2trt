import glob
import os
import os.path as osp
from xml.etree.ElementInclude import include
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.dirname(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob("*.cpp") + glob.glob("*.cu")

setup(name="customOp_cuda",
      version="1.0",
      author="wn",
      description="cuda torch extension op(Linear)",
      ext_modules=[
            CUDAExtension(
                  name="customOp_cuda", 
                  sources=["customCudaOp.cpp", "customCudaOpKernelAccessor.cu"],
                  include_dirs=include_dirs,
                  extra_compile_args={
                        "CXX": ["-O2"],
                        "nvcc": ["-O2"]
                  }
                  )
                  ],
      cmdclass={
            "build_ext": BuildExtension
            })
