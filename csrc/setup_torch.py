from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="storage_offload_ext",
    ext_modules=[
        CUDAExtension(
            "storage_offload_ext",
            ["storage_offload_ext.cu"],
            include_dirs=['/usr/include'],
            library_dirs=['/usr/lib/x86_64-linux-gnu'],
            libraries=['uring'],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": ["-O3", "-std=c++17"]
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
