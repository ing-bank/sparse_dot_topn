import os
import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


here = os.path.abspath(os.path.dirname(__file__))
# Get the long description from the README file
with open(os.path.join(here, 'README.md')) as f:
    long_description = f.read()


if os.name == 'nt':
    extra_compile_args = ["-Ox"]
else:
    extra_compile_args = ['-std=c++0x', '-pthread', '-O3']

original_ext = Extension('sparse_dot_topn.sparse_dot_topn',
                         sources=['./sparse_dot_topn/sparse_dot_topn.pyx',
                                  './sparse_dot_topn/sparse_dot_topn_source.cpp'],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=extra_compile_args,
                         language='c++')

threaded_ext = Extension('sparse_dot_topn.sparse_dot_topn_threaded',
                         sources=[
                             './sparse_dot_topn/sparse_dot_topn_threaded.pyx',
                             './sparse_dot_topn/sparse_dot_topn_source.cpp',
                             './sparse_dot_topn/sparse_dot_topn_parallel.cpp'],
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=extra_compile_args,
                         language='c++')


setup(
    name='sparse_dot_topn',
    version='0.2.6',
    description='This package boosts a sparse matrix multiplication '\
                'followed by selecting the top-n multiplication',
    keywords='cosine-similarity sparse-matrix scipy cython',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ing-bank/sparse_dot_topn',
    author='Zhe Sun',
    author_email='ymwdalex@gmail.com',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
        'numpy'
    ],
    packages=['sparse_dot_topn'],
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([original_ext, threaded_ext]),
)

