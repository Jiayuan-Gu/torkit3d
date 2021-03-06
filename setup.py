import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

import torkit3d


def get_extensions():
    """Refer to torchvision."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torkit3d", "csrc")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    sources = main_file
    extension = CppExtension

    define_macros = []
    extra_compile_args = {}
    if (torch.cuda.is_available() and (CUDA_HOME is not None)) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = ["-O2"]
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args = {
            "cxx": [],
            "nvcc": nvcc_flags,
        }

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir, os.path.join(extensions_dir, "include")]
    print("sources:", sources)

    ext_modules = [
        extension(
            "torkit3d._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules


setup(
    name="torkit3d",
    version=torkit3d.__version__,
    author="Jiayuan Gu",
    author_email="jigu@eng.ucsd.edu",
    description="Pytorch Toolkit3D",
    install_requires=[
        "torch>=1.5.1",
        "numpy",
        "scipy",
    ],
    python_requires=">=3.6",
    url="",
    packages=find_packages(exclude=["tests"]),
    long_description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
