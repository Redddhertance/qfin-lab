from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

#the permutation backend is the one bit of this project that isn't python. building it through
#pybind11's helpers means `pip install .` compiles gaka_core in place, no manual c++ command and
#no checked-in .so that only works on my machine.
ext_modules = [
    Pybind11Extension(
        'gaka_core',
        ['src/gaka_core.cpp'],
        cxx_std=14,
    ),
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
