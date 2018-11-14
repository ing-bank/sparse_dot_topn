import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_utils = Extension('sparse_dot_topn.sparse_dot_topn',
                      sources=['./sparse_dot_topn/sparse_dot_topn.pyx', './sparse_dot_topn/sparse_dot_topn_source.cpp'],
                      include_dirs=[numpy.get_include()],
                      #libraries=[],
                      extra_compile_args=['-std=c++0x', '-Os'],
                      language='c++',
                     )

setup(name='sparse_dot_topn',
      version='0.2',
      description='This package boosts a sparse matrix multiplication '\
                  'followed by selecting the top-n multiplication',
      keywords='cosine-similarity sparse-matrix scipy cython',
      setup_requires=[
          # Setuptools 18.0 properly handles Cython extensions.
          'setuptools>=18.0',
          'cython',
          'numpy'
      ],
      packages=['sparse_dot_topn'],
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize([ext_utils]),
     )

