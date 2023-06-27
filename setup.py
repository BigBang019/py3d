from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import glob
import os

eigen_headers = "-I/usr/include/eigen3"

headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

smooth_sources = glob.glob("wrapper/smooth/*.cpp") + glob.glob("wrapper/smooth/*.cu")
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

arap_sources = glob.glob("wrapper/arap/*.cpp") + glob.glob("wrapper/arap/*.cu")
arap_sources.remove("wrapper/arap/main.cu")
arap_headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "arap")

arap = CUDAExtension(
    name = "py3d.arap",
    sources = arap_sources,
    extra_compile_args={
        "cxx": ["-O2",
                arap_headers,
                headers,
                eigen_headers
                ],
        "nvcc": ["-O2", arap_headers,
                 headers,
                 eigen_headers
                 ]
    },
)

cgal_sources = glob.glob("wrapper/cgal/*.cpp") + glob.glob("wrapper/cgal/*.cu")
cgal_sources.remove("wrapper/cgal/main.cpp")
cgal_headers = "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "cgal")

cgal = CUDAExtension(
    name = "py3d.cgal",
    sources = cgal_sources,
    extra_compile_args={
        "cxx": ["-O2", cgal_headers,
                headers,
                eigen_headers
                ],
        "nvcc": ["-O2", cgal_headers,
                 headers,
                 eigen_headers
                 ]
    },
)

setup(
    name='py3d',
    ext_modules = [
        smooth, arap, cgal
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)