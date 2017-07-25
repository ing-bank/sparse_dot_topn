import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, Extension

from cossim_topn import __version__

ext_utils = Extension('cossim_topn.cossim_topn',
                      sources=['./cossim_topn/cossim_topn.pyx', './cossim_topn/cossim_topn_source.cpp'],
                      include_dirs=[numpy.get_include()],
                      #libraries=[],
                      extra_compile_args=['-std=c++0x', '-Os'],
                      language='c++',
                     )

setup(name='cossim_topn',
      version=__version__,
      setup_requires=[
          # Setuptools 18.0 properly handles Cython extensions.
          'setuptools>=18.0',
          'cython',
      ],
      packages=['cossim_topn'],
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize([ext_utils]),
     )
