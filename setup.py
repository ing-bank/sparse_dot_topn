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
            # got error `'dict' object has no attribute '__NUMPY_SETUP__'`
            # Follow this solution https://github.com/SciTools/cf-units/blob/master/setup.py#L99
            def _set_builtin(name, value):
                if isinstance(__builtins__, dict):
                    __builtins__[name] = value
                else:
                    setattr(__builtins__, name, value)

            _build_ext.finalize_options(self)
            # Prevent numpy from thinking it is still in its setup process:
            _set_builtin('__NUMPY_SETUP__', False)
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

array_wrappers_ext = Extension('sparse_dot_topn.array_wrappers',
                         sources=[
                                    './sparse_dot_topn/array_wrappers.pyx',
                                ],
                         extra_compile_args=extra_compile_args,
                         language='c++')

original_ext = Extension('sparse_dot_topn.sparse_dot_topn',
                         sources=[
                                    './sparse_dot_topn/sparse_dot_topn.pyx',
                                    './sparse_dot_topn/sparse_dot_topn_source.cpp'
                                ],
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
    version='0.3.0',
    description='This package boosts a sparse matrix multiplication '\
                'followed by selecting the top-n multiplication',
    keywords='cosine-similarity sparse-matrix scipy cython',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ing-bank/sparse_dot_topn',
    author='Zhe Sun',
    author_email='ymwdalex@gmail.com',
    license='Apache 2.0',
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=42',
        'cython>=0.29.15',
        'numpy>=1.16.6', # select this version for Py2/3 compatible
        'scipy>=1.2.3'   # select this version for Py2/3 compatible
    ],
    install_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=42',
        'cython>=0.29.15',
        'numpy>=1.16.6', # select this version for Py2/3 compatible
        'scipy>=1.2.3'   # select this version for Py2/3 compatible
    ],
    zip_safe=False,
    packages=find_packages(),
    cmdclass={'build_ext': my_build_ext},
    ext_modules=[array_wrappers_ext, original_ext, threaded_ext],
    package_data = {
        'sparse_dot_topn': ['./sparse_dot_topn/*.pxd']
    },
    include_package_data=True,    
)