# flake8: noqa
import os
from setuptools import setup, Extension, find_packages

# workaround for numpy and Cython install dependency
# the solution is from https://stackoverflow.com/a/54138355
def my_build_ext(pars):
    # import delayed:
    from setuptools.command.build_ext import build_ext as _build_ext
    class build_ext(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    #object returned:
    return build_ext(pars)


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
                         extra_compile_args=extra_compile_args,
                         language='c++')

threaded_ext = Extension('sparse_dot_topn.sparse_dot_topn_threaded',
                         sources=[
                             './sparse_dot_topn/sparse_dot_topn_threaded.pyx',
                             './sparse_dot_topn/sparse_dot_topn_source.cpp',
                             './sparse_dot_topn/sparse_dot_topn_parallel.cpp'],
                         extra_compile_args=extra_compile_args,
                         language='c++')


setup(
    name='sparse_dot_topn',
    version='0.2.9',
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
        'cython>=0.29.15',
        'numpy>=1.16.6', # select this version for Py2/3 compatible
        'scipy>=1.2.3'   # select this version for Py2/3 compatible
    ],
    install_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython>=0.29.15',
        'numpy>=1.16.6', # select this version for Py2/3 compatible
        'scipy>=1.2.3'   # select this version for Py2/3 compatible
    ],
    packages=find_packages(),
    cmdclass={'build_ext': my_build_ext},
    ext_modules=[original_ext, threaded_ext],
)
