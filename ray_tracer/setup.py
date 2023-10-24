from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("ray_tracer/engine/intersect.pyx"),
    package_dir={'ray_tracer': 'ray_tracer'},
    include_dirs=[numpy.get_include()]
)