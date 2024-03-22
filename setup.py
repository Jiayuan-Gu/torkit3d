"""
References:
- https://github.com/NVIDIA/apex: `--no-build-isolation` if pyproject.toml is used. (PEP 518)
- https://github.com/facebookresearch/pytorch3d: use `runpy` to read verison info. use `setup.cfg` to configure isort instead of `pyproject.toml`.
- https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html: MANIFEST.in
- https://github.com/erikwijmans/Pointnet2_PyTorch
- https://github.com/haosulab/SAPIEN
- https://github.com/NVlabs/nvdiffrast: how to compile once actually used.
"""

import glob
import os
import runpy

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


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
        if "NVCC_FLAGS" in os.environ:
            nvcc_flags = os.environ["NVCC_FLAGS"].split(" ")
        else:
            nvcc_flags = []
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


def get_version():
    version = runpy.run_path("torkit3d/version.py")
    return version["__version__"]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torkit3d",
    version=get_version(),
    author="Jiayuan Gu",
    author_email="jigu@ucsd.edu",
    url="https://github.com/Jiayuan-Gu/torkit3d",
    description="Pytorch Toolkit3D",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch>=1.12.1",
        "numpy",
    ],
    extras_require={"dev": ["pytest", "isort", "black"]},
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests"]),
    # include_package_data=True,
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
