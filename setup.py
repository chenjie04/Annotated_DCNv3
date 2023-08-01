import glob
import os
import torch

from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

from setuptools import setup, find_packages

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None):
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError("Cuda is not available")
    
    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension("DCNv3", # 这里是模块名，和 PYBIND11_MODULE 中m.def()定义的算子名共同决定了调用算子的方式，如：DCNv3.forward()，DCNv3.backward()
                  sources, 
                  include_dirs=include_dirs, 
                  define_macros=define_macros, 
                  extra_compile_args=extra_compile_args,
                  )
    ]

    return ext_modules


setup(
    name="DCNv3",
    version="1.0.1",
    url="https://github.com/OpenGVLab/InternImage",
    description="PyTorch Wrapper for CUDA Functions of DCNv3",
    packages=find_packages(exclude=(
        "configs",
        "tests",
    )),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)