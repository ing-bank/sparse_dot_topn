# Release history:

## v1.1.5

### Changes

BLD: Set rpath for libomp on MacOS to fix compatibility with latest XGBoost version

## v1.1.4

### Changes

PKG: Suppress error for double initialisation of OpenMP

## v1.1.3

### Changes

BLD: Unrestrict nanobind by @RUrlus in #112

### Internal

FIX: [C++] Remove usage of nb::raw_doc

## v1.1.2

### Changes

BLD: Restrict nanobind version to <2.0 by @RUrlus in #111

### Internal

CICD: Add trusted publishing and updated MacOS runners by @RUrlus in #108
MAINT: Correct name typo in readme by @mmtevelde in #110
CICD: Bump pypa/cibuildwheel from 2.18.0 to 2.18.1 by @dependabot in #109

## v1.1.1

### Internal

- FIX: Prevent Scipy from dropping columns that are all zero for sub-matrices

## v1.1.0

Add new function to select top-n from blocks of a sparse matrix matmul.
Function will return a zipped matrix Z, where Z = [sorted top n results > lower_bound for each row of C_j], where C_j = A.dot(B_j) and where B has been split row-wise into sub-matrices B_j.

### API

- ENH: new function zip_sp_matmul_topn can zip matrices zip_j A.dot(B_j)

## v1.0.0

This introduces major and potentially breaking changes to the API.
Please see the migration guide in the README for details.

### API

- API: `awesome_cossim_topn` is superseded with `sp_matmul_topn`.
- API: `awesome_cossim_topn` has been deprecated and will be removed in a future version.
- API: `ntop` parameter has been renamed to `topn`
- API: `lower_bound` parameter has been renamed to `threshold`
- API: `use_threads` and `n_jobs` parameters have been combined into `n_threads`
- API: `return_best_ntop` parameter has been removed
- API: `test_nnz_max` parameter has been removed
- API: default parameter value for `threshold` changed from `0.0` to `None` (disabled)
- API: default parameter value for `sort` changed to `False`

- ENH: Add support for 32 and 64bit integers
- BLD: Add support for CPython 3.12

### Internal

- BLD: Switch to pyproject.toml based setup (scikit-build-core)
- FIX: [C++] Resolve unneeded memory allocation that solved hidden buffer-overrun in multithreaded implementation
- BLD: [C++] Switch to Nanobind bindings
- CHG: [C++] Switch to OpenMP for multithreading
- ENH: [C++] Use MaxHeap to collect top-n results over vector of candidates

## v0.3.6
- Adds support for Cython >= 3.0

## v0.3.5
- Restrict Cython version to <3.0

## v0.3.4
- Add Python 3.11 wheels
- Fix a compilation error when std:: is missing

## v0.3.3
- Upgrade to 0.3.3, since PyPI had versioning problem when we release 0.3.2

## v0.3.2
- Fix the Numpy ABI compatibility issues [issue-48](https://github.com/ing-bank/sparse_dot_topn/issues/48) and all related issues
- Add Github Actions to build Python release in Linux, MacOS and Windows automatically

## v0.3.1
- Adding the possibility to use smaller data type float32 instead of the default float64
- Adding unit tests to cover both data types
- Adding unit tests to cover return_best_ntop==False

## v0.3.0
- defragmented memory used during computation [PR-53](https://github.com/ing-bank/sparse_dot_topn/pull/53)
- Enable Github Action for unit test
- add license metadata [PR-47](https://github.com/ing-bank/sparse_dot_topn/pull/47)

## v0.2.9
- added unit tests. 
- awesome_cossim_topn checks for zero input matrices.
- fix flake8 warnings.

## v0.2.8
- quick fix for nt installation problem [ISSUE-26]

## v0.2.7
- add n_jobs value validation
- matrix dimension validation
- change windows mingw32 compiler
- fix numpy and Cython install dependency

## v0.2.6
- import parallel implementation

## v0.2.5
- fix python 2 install problem

## v0.2.5
- fix python 2 install problem, but failed

## v0.2.3
- Make long description use markdown format

## v0.2.2
- update readme file name

## v0.2.1
- fix the problem when pip install by tar.gz;
- update meta-information of package

## v0.2, Nov 14, 2018:
- improve the function import layer. Now awesome_cossim_topn can be used directly after `from sparse_dot_topn import awesome_cossim_topn`.
- improve the module installation

## v0.1, July 13, 2017:
- Release the first version

