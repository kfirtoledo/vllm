from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="storage_offload_ext",
    ext_modules=[
        CUDAExtension(
            "storage_offload_ext",
            ["storage_offload_ext.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": ["-O3", "-std=c++17"]
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
