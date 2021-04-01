from setuptools import setup, find_packages

setup(
    name="torkit3d",
    version="0.1.0",
    author="Jiayuan Gu",
    author_email="jigu@eng.ucsd.edu",
    description="Pytorch Toolkit3D",
    install_requires=[
        'torkit',
        'torch==1.5.1',
        'numpy',
        'scipy',
        'open3d==0.11.2',
    ],
    python_requires='>=3.6',
    url="",
    packages=find_packages(exclude=("tests",)),
    long_description="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
