from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="customOp_cpp",
      ext_modules=[cpp_extension.CppExtension("customOp_cpp", ["customOp.cpp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})

