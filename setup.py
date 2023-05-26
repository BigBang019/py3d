from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import glob
import os

eigen_headers = "-I/usr/include/eigen3"

headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "core")

smooth_sources =  glob.glob("wrapper/smooth/*.cpp") + glob.glob("wrapper/smooth/*.cu")
smooth_sources.remove("wrapper/smooth/main.cu")
smooth_headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "smooth")
print(smooth_sources)

smooth = CUDAExtension(
    name = "py3d.smooth",
    sources = smooth_sources,
    extra_compile_args={
        "cxx": ["-O2",
                smooth_headers,
                headers,
                eigen_headers
                ],
        "nvcc": ["-O2", smooth_headers,
                 headers,
                 eigen_headers
                 ]
    },
)

setup(
    name='py3d',
    ext_modules = [
        smooth
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)