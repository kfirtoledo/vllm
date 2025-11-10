from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="storage_offload",
    ext_modules=[
        CUDAExtension(
            "storage_offload",
             sources=[
                "storage_offload.cu",
            ],
            libraries=['nvidia-ml', 'numa', 'cuda'],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fopenmp"],
                "nvcc": ["-O3", "-std=c++17", "-Xcompiler", "-std=c++17","-Xcompiler", "-fopenmp"]
            }
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
