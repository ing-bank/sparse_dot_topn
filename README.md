# sparse\_dot\_topn

[![MacOS](https://github.com/ing-bank/sparse_dot_topn/actions/workflows/macos.yml/badge.svg)](https://github.com/ing-bank/sparse_dot_topn/actions/workflows/macos.yml)
[![Linux](https://github.com/ing-bank/sparse_dot_topn/actions/workflows/linux.yml/badge.svg)](https://github.com/ing-bank/sparse_dot_topn/actions/workflows/linux.yml)
[![Windows](https://github.com/ing-bank/sparse_dot_topn/actions/workflows/windows.yml/badge.svg)](https://github.com/ing-bank/sparse_dot_topn/actions/workflows/windows.yml)
[![License](https://img.shields.io/github/license/ing-bank/sparse_dot_topn)](https://github.com/ing-bank/sparse_dot_topn/blob/master/LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)


[![Release_date](https://img.shields.io/github/release-date/ing-bank/sparse_dot_topn)](https://github.com/ing-bank/sparse_dot_topn/releases)
[![PyPi](https://img.shields.io/pypi/v/sparse-dot-topn.svg)](https://pypi.org/project/sparse-dot-topn/)
[![Downloads](https://pepy.tech/badge/sparse_dot_topn)](https://pepy.tech/project/sparse_dot_topn)

**sparse\_dot\_topn** provides a fast way to performing a sparse matrix multiplication followed by top-n multiplication result selection.

Comparing very large feature vectors and picking the best matches, in practice often results in performing a sparse matrix multiplication followed by selecting the top-n multiplication results.

**sparse\_dot\_topn** provides a (parallelised) sparse matrix multiplication implementation that integrates selecting the top-n values, resulting in a significantly lower memory footprint and improved performance.
On Apple M2 Pro over two 20k x 193k TF-IDF matrices **sparse\_dot\_topn** can be up to 6 times faster when retaining the top 10 values per row and utilising 8 cores.
See the benchmark directory for details.

## Usage

`sp_matmul_topn` supports `{CSR, CSC, COO}` matrices with `{32, 64}bit {int, float}` data.
Note that `COO` and `CSC` inputs are converted to the `CSR` format and are therefore slower.
Two options to further reduce memory requirements are `threshold` and `density`.
Optionally, the values can be sorted such that the first column for a given row contains the largest value.
Note that `sp_matmul_topn(A, B, top_n=B.shape[1])` is equal to `sp_matmul(A, B)` and `A.dot(B)`.

**If you are migrating from `v0.*` please see the migration guide below for details.**

```python
import scipy.sparse as sparse
from sparse_dot_topn import sp_matmul, sp_matmul_topn

A = sparse.random(1000, 100, density=0.1, format="csr")
B = sparse.random(100, 2000, density=0.1, format="csr")

# Compute C and retain the top 10 values per row
C = sp_matmul_topn(A, B, top_n=10)

# or paralleslised matrix multiplication without top-n selection
C = sp_matmul(A, B, n_threads=2)
# or with top-n selection
C = sp_matmul_topn(A, B, top_n=10, n_threads=2)

# If you are only interested in values above a certain threshold
C = sp_matmul_topn(A, B, top_n=10, threshold=0.8)

# If you set the threshold we cannot easily determine the number of non-zero
# entries beforehand. Therefore, we allocate memory for `ceil(top_n * A.shap[0] * density)`
# non-zero entries. You can set the expected density to reduce the amount pre-allocated
# entries. Note that if we allocate too little an expensive copy(ies) will need to hapen.
C = sp_matmul_topn(A, B, top_n=10, threshold=0.8, density=0.1)
```

## Installation

**sparse\_dot\_topn** provides wheels for CPython 3.8 to 3.12 for:

* Windows (64bit)
* Linux (64bit)
* MacOS (x86 and ARM)

```shell
pip install sparse_dot_topn
```

**sparse\_dot\_topn** relies on a C++ extension for the computationally intensive multiplication routine.
**Note that the wheels vendor/ships OpenMP with the extension to provide parallelisation out-of-the-box.**
**This may cause issues when used in combination with other libraries that ship OpenMP like PyTorch.**
If you run into any issues with OpenMP see INSTALLATION.md for help or run the function without specifying the `n_threads` argument.

Installing from source requires a C++17 compatible compiler.
If you have a compiler available it is advised to install without the wheel as this enables architecture specific optimisations.

You can install from source using:

```shell
pip install sparse_dot_topn --no-binary sparse_dot_topn
```

### Build configuration

**sparse\_dot\_topn** provides some configuration options when building from source.
Building from source can enable architecture specific optimisations and is recommended for those that have a C++ compiler installed.
See INSTALLATION.md for details.

## Distributing the top-n multiplication of two large O(10M+) sparse matrices over a cluster

The top-n multiplication of two large O(10M+) sparse matrices can be broken down into smaller chunks.
For example, one may want to split sparse matrices into matrices with just 1M rows, and do the
the (top-n) multiplication of all those matrix pairs.
Reasons to do this are to reduce the memory footprint of each pair, and to employ available distributed computing power.

The pairs can be distributed and calculated over a cluster (eg. we use a spark cluster).
The resulting matrix-products are then zipped and stacked in order to reproduce the full matrix product.

Here's an example how to do this, where we are matching 1000 rows in sparse matrix A against 600 rows in sparse matrix B,
and both A and B are split into chunks.

```python
import numpy as np
import scipy.sparse as sparse
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn

# 1a. Example matching 1000 rows in sparse matrix A against 600 rows in sparse matrix B.
A = sparse.random(1000, 2000, density=0.1, format="csr", dtype=np.float32, random_state=rng)
B = sparse.random(600, 2000, density=0.1, format="csr", dtype=np.float32, random_state=rng)

# 1b. Reference full matrix product with top-n
C_ref = sp_matmul_topn(A, B.T, top_n=10, threshold=0.01, sort=True)

# 2a. Split the sparse matrices. Here A is split into three parts, and B into five parts.
As = [A[i*200:(i+1)*200] for i in range(5)]
Bs = [B[:100], B[100:300], B[300:]]

# 2b. Perform the top-n multiplication of all sub-matrix pairs, here in a double loop.
# E.g. all sub-matrix pairs could be distributed over a cluster and multiplied there.
Cs = [[sp_matmul_topn(Aj, Bi.T, top_n=10, threshold=0.01, sort=True) for Bi in Bs] for Aj in As]

# 2c. top-n zipping of the C-matrices, done over the index of the B sub-matrices.
Czip = [zip_sp_matmul_topn(top_n=10, C_mats=Cis) for Cis in Cs]

# 2d. stacking over zipped C-matrices, done over the index of the A sub-matrices
# The resulting matrix C equals C_ref.
C = sparse.vstack(Czip, dtype=np.float32)
```

## Migrating to v1.

**sparse\_dot\_topn** v1 is a significant change from `v0.*` with a new bindings and API.
The new version adds support for CPython 3.12 and now supports both ints as well as floats.
Internally we switched to a max-heap to collect the top-n values which significantly reduces memory-footprint.
The former implementation had `O(n_columns)` complexity for the top-n selection where we now have `O(top-n)` complexity.
**`awesome_cossim_topn` has been deprecated and will be removed in a future version.**

Users should switch to `sp_matmul_topn` which is largely compatible:

For example:

```python
C = awesome_cossim_topn(A, B, ntop=10)
```

can be replicated using:

```python
C = sp_matmul_topn(A, B, top_n=10, threshold=0.0, sort=True)
```

### API changes
1. `ntop` has been renamed to `topn`
2. `lower_bound` has been renamed to `threshold`
3. `use_threads` and `n_jobs` have been combined into `n_threads`
4. `return_best_ntop` option has been removed
5. `test_nnz_max` option has been removed
6. `B` is auto-transposed when its shape is not compatible but its transpose is.

The output of `return_best_ntop` can be replicated with:

```python
C = sp_matmul_topn(A, B, top_n=10)
best_ntop = np.diff(C.indptr).max()
```

### Default changes

1. `threshold` no longer `0.0` but disabled by default

This enables proper functioning for matrices that contain negative values.
Additionally a different data-structure is used internally when collecting non-zero results that has a much lower memory-footprint than previously.
This means that the effect of the `threshold` parameter on performance and memory requirements is negligible. 
If the `threshold` is `None` we pre-compute the number of `non-zero` entries, this can significantly reduce the required memory at a mild (~10%) performance penalty.

2. `sort = False`, the result matrix is no longer sorted by default

The matrix is returned with the same column order as if not filtering of the top-n results has taken place.
This means that when you set `top_n` equal to the number of columns of `B` you obtain the same result as normal multiplication,
i.e. `sp_matmul_topn(A, B, top_n=B.shape[1])` is equal to `A.dot(B)`.

## Contributing

Contributions are very welcome, please see CONTRIBUTING for details.

### Contributors

This package was developed and is maintained by authors (previously) affiliated with ING Analytics Wholesale Banking Advanced Analytics.
The original implementation was based on modified version of Scipy's CSR multiplication implementation.
You can read about it in a [blog](https://medium.com/@ingwbaa/https-medium-com-ingwbaa-boosting-selection-of-the-most-similar-entities-in-large-scale-datasets-450b3242e618) [(mirror)](https://www.sun-analytics.nl/posts/2017-07-26-boosting-selection-of-most-similar-entities-in-large-scale-datasets/) written by Zhe Sun.

* [Zhe Sun](https://github.com/ymwdalex/)
* [Ahmet Erdem](https://github.com/aerdem4)
* [Stephane Collot](https://github.com/stephanecollot)
* [Particular Miner](https://github.com/ParticularMiner) (no ING affiliation)
* [Ralph Urlus](https://github.com/RUrlus)
* [Max Baak](https://github.com/mbaak)
