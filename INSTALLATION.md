## Installation

**sparse\_dot\_topn** provides wheels for CPython 3.10 to 3.14 for:

* Windows (64bit)
* Linux (64bit)
* MacOS (ARM)

On other platforms, such as MacOS on Intel, pip builds the package from source.

```shell
pip install sparse_dot_topn
```

**sparse\_dot\_topn** relies on a C++ extension for the computationally intensive multiplication routine.
Note that the wheels vendor/ships OpenMP with the extension to provide parallelisation out-of-the-box.

### Building from source

Installing from source requires a C++17 compatible compiler.
If you have a compiler available it is advised to install without the wheel as this enables architecture specific optimisations.

You can install from source using:

```shell
pip install sparse_dot_topn --no-binary sparse_dot_topn
```

### Native builds

When you're building from source we assume that the target architecture is the one you are building on, aka a native build.
This generally results in faster binaries at the cost that they cannot be used on different systems.
Native architecture flags are enabled and the CPU is checked for support of SSE2, SSE4, AVX and AX2.

If you are building from source for a different system, you can disable this with:

```shell
CMAKE_ARGS="-DSDTN_ENABLE_ARCH_FLAGS=OFF" pip install sparse_dot_topn --no-binary sparse_dot_topn
```

### Multithreading

Parallelisation is supported and automatically enabled if OpenMP can be found.

You can either explicitly enable OpenMP, which will now raise an exception if it cannot be found:

```shell
CMAKE_ARGS="-DSDTN_ENABLE_OPENMP=ON" pip install sparse_dot_topn --no-binary sparse_dot_topn
```
or explicitly disable it:

```shell
CMAKE_ARGS="-DSDTN_DISABLE_OPENMP=ON" pip install sparse_dot_topn --no-binary sparse_dot_topn
```

#### Finding OpenMP

If OpenMP cannot be found you can pass it's root directory using the `OpenMP_ROOT` definition.
For example, for Homebrew users on Apple silicon: 

```shell
CMAKE_ARGS="-DSDTN_ENABLE_OPENMP=ON -DOpenMP_ROOT=$(brew --prefix)/opt/libomp" \
pip install sparse_dot_topn --no-binary sparse_dot_topn
```

